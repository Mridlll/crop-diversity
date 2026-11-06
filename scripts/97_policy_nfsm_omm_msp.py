"""
97_policy_nfsm_omm_msp.py

USAGE
    C:/Users/mridu/anaconda3/envs/geo_env/python.exe scripts/97_policy_nfsm_omm_msp.py

Assembles three public policy and price series for the crop-group work, every row
carrying the URL of the document it came from:

  (1) National Food Security Mission (NFSM) district coverage, by component and year
      2007-08 to 2019-20: rice, wheat, pulses, coarse cereals, nutri-cereals.
  (2) Odisha Millets Mission (now Shree Anna Abhiyan): blocks and districts by year of
      entry from 2017-18.
  (3) Minimum Support Prices by crop and crop year, 1997-98 to 2020-21.

Source PDFs were downloaded to outputs/crop_groups/raw_sources/ (URLs in SOURCES
below). District lists in the 2007 and 2009 NFSM guidelines are typeset in a layout
that text extraction scrambles, so they are transcribed by hand below from the
annexure pages, and the per-state counts printed in the PDFs are asserted. The
2014-15 (XII Plan) and 2018-19 lists are parsed from word positions with pdfplumber,
and again every state count printed in the PDF is asserted.

District names are kept as printed. Each is then matched to the crop panel
(outputs/crop_diversity_analysis/district_diversity_indices_corrected.csv, 725
districts, of which 606 are the analysis sample) by: alias table -> exact name within
state -> fuzzy (rapidfuzz token_sort_ratio >= 85) within state. Unmatched names are
written out, never dropped.

Outputs (outputs/crop_groups/):
  policy_nfsm_district_lists.csv        one row per (component, vintage, listed district)
  policy_nfsm_state_rules.csv           components/years where coverage is "all districts
                                        of state S" rather than a list
  policy_nfsm_district_year.csv         panel district x year x component, 0/1 + basis
  policy_nfsm_unmatched.csv
  policy_omm_blocks.csv                 block, district, entry year, source
  policy_omm_district_year.csv          panel district x year: OMM blocks (cumulative)
  policy_msp_crop_year.csv              tidy MSP, one row per crop x variety x crop year
  policy_msp_crop_group_map.csv         MSP crop -> crop group
  policy_msp_rbi_crosscheck.csv         RBI Handbook series vs the compiled series
"""
import re
import unicodedata
import numpy as np
import pandas as pd
import pdfplumber
from rapidfuzz import fuzz, process

REPO = r"D:/crop-diversity"
OUT = REPO + "/outputs/crop_groups"
RAW = OUT + "/raw_sources"
DIV = REPO + "/outputs/crop_diversity_analysis"
COV = REPO + "/outputs/shrug_covariates"

SOURCES = {
    "nfsm_2007": ("NFSM Operational Guidelines, August 2007 (Annexure II a-c)",
                  "https://agritech.tnau.ac.in/govt_schemes_services/govt_schemes_National%20Food%20Security%20Mission.pdf"),
    "nfsm_2009": ("NFSM Operational Guidelines, revised 2009 (Annexure II a-c)",
                  "https://www.nfsm.gov.in/Guidelines/ContentE.pdf"),
    "nfsm_xii": ("NFSM Operational Guidelines, 12th Five Year Plan (Annexure II a-d), in force from 2014-15",
                 "https://www.nfsm.gov.in/Guidelines/NFSMXII.pdf"),
    "nfsm_2018": ("NFSM Operational Guidelines 2018-19 to 2019-20 (Annexure II a-d, VIII a)",
                  "https://www.nfsm.gov.in/Guidelines/Guideline_nfsmandoilseed201819to201920.pdf"),
    "nfsm_2010_expansion": ("PIB-derived note, 8 Mar 2011: NFSM in 476 districts of 17 states; pulses in 467 "
                            "districts of 16 states incl. 10 of Assam and 15 of Jharkhand",
                            "https://currentaffairsappsc.blogspot.com/2011/03/476-districts-covered-under-nfsm.html"),
    "nfsm_csd": ("CSD, Role of NFSM in improving agricultural productivity (since 2010-11: 468 pulses districts "
                 "in 16 states; all districts of the 14 ISOPOM states)",
                 "http://www.csdindia.org/pdfs/Project-reports/ROLE%20OF%20NATIONAL%20FOOD%20SECURITY%20MISSION%20(NFSM)%20IN%20IMPROVING%20AGRICULTURAL%20PRODUCTIVITY.pdf"),
    "omm_ar_1718": ("OMM Annual Report 2017-18, Annexure 2 (first-phase blocks)",
                    "https://milletsodisha.com/uploads/files/publication/Annual-Report-Odisha-Millets-Mission_2017-18_24.09.2019.pdf"),
    "omm_ar_2021": ("OMM Annual Report 2020-21, phase-wise scale-up table (blocks by district and year)",
                    "https://milletsodisha.com/uploads/files/Odisha%20Millets%20Mission%20-%20Annual%20Report%202020-21.pdf"),
    "omm_guideline_2226": ("OMM Operational Guideline 2022-2026, section 3.1 (84 existing + 58 new blocks)",
                           "https://milletsodisha.com/uploads/files/guidelinedocument/Approved%20OMM%20Guideline_Final.pdf"),
    "omm_modifications": ("Approved modifications of OMM guideline, Annexure-I (34 new blocks, 142 to 177)",
                          "https://milletsodisha.com/uploads/files/guidelinedocument/Approved_MODIFICATIONS%20OF%20ODISHA%20MILLETS%20MISSION%20GUIDELINE.pdf"),
    "es_9900": ("Economic Survey 1999-2000, Statistical Appendix Table 5.5 (crop-year basis)",
                "https://www.indiabudget.gov.in/budget_archive/es99-2000/app5.5.pdf"),
    "des_2004": ("DAC/DES, Minimum Support Prices (according to crop year), as on 03.11.2004",
                 "https://desagri.gov.in/wp-content/uploads/2021/04/MINIMUM-SUPPORT-PRICES03-11-2004.pdf"),
    "es_1213": ("Economic Survey 2012-13, Statistical Appendix Table 5.5 (crop-year basis)",
                "https://www.indiabudget.gov.in/budget_archive/es2012-13/estat1.pdf"),
    "cacp_1221": ("CACP, MSPs recommended by CACP and fixed by Government, 2012-13 to 2021-22 (crop year)",
                  "https://www.spuvvn.edu/academic_centres/c_c_s/MSP_SINCE_2012_13.pdf"),
    "rbi_t23": ("RBI Handbook of Statistics on Indian Economy 2025-26, Table 23 (MSP foodgrains, crop year)",
                "https://www.rbi.org.in/scripts/PublicationsView.aspx?id=23848"),
    "rbi_t24": ("RBI Handbook of Statistics on Indian Economy 2025-26, Table 24 (MSP non-foodgrains, crop year)",
                "https://www.rbi.org.in/scripts/PublicationsView.aspx?id=23849"),
}

YEARS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2007, 2020)]   # 2007-08 .. 2019-20


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = s.upper().replace("&", " AND ")
    # keep parentheticals that carry the identity of the district (direction, rural/urban)
    s = re.sub(r"\(\s*(E|W|N|S|EAST|WEST|NORTH|SOUTH|DEHAT|RURAL|URBAN|NAGAR)\s*\)", r" \1 ", s)
    s = re.sub(r"\(.*?\)", " ", s)          # drop other parentheticals (old names, HQ towns)
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    s = re.sub(r"\b(DISTRICT|DIST)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------- crop panel names
ind = pd.read_csv(DIV + "/district_diversity_indices_corrected.csv")
fp = pd.read_csv(COV + "/final_panel.csv")
S606 = set(fp.loc[fp["in_final"] == True, "div_key"])
xw = pd.read_csv(COV + "/district_crosswalk.csv")[["div_key", "shrug_key"]]
panel = ind[["state_name", "district_name", "district_key"]].copy()
panel["in_606"] = panel["district_key"].isin(S606)
panel = panel.merge(xw, left_on="district_key", right_on="div_key", how="left").drop(columns="div_key")
panel = panel.rename(columns={"shrug_key": "parent2011_key"})
panel["nd"] = panel["district_name"].map(norm)

# source state name -> panel states searched (pre-2014 Andhra lists include Telangana)
STATE_MAP = {
    "ANDHRA PRADESH": ["ANDHRA PRADESH", "TELANGANA"], "ANDHRA PRADSH": ["ANDHRA PRADESH", "TELANGANA"],
    "TELANGANA": ["TELANGANA", "ANDHRA PRADESH"], "TELANANA": ["TELANGANA", "ANDHRA PRADESH"],
    "MADYA PRADESH": ["MADHYA PRADESH"], "UTTRAKHAND": ["UTTARAKHAND"], "ORISSA": ["ODISHA"], "ODISHA": ["ODISHA"],
    "CHHATTISHGARH": ["CHHATTISGARH"], "CHHATTISGARH": ["CHHATTISGARH"],
    "TAMILNADU": ["TAMIL NADU"], "TAMIL NADU": ["TAMIL NADU"],
    "JAMMU AND KASHMIR": ["JAMMU AND KASHMIR", "LADAKH"],
}

# printed name -> panel name (normalised), applied within state; built by inspecting
# the unmatched list. Renamings and old spellings only, never a different district.
ALIAS = {
    "VIZIANAGARM": "VIZIANAGARAM", "MAHABOOBNAGAR": "MAHABUBNAGAR", "MAHBUBNAGAR": "MAHABUBNAGAR",
    "NELLORE": "SRI POTTI SRIRAMULU NELLORE", "CUDDAPAH": "KADAPA", "ANANTPUR": "ANANTAPUR",
    "VISAKHAPATNAM": "VISAKHAPATANAM", "VISHAKHAPATNAM": "VISAKHAPATANAM",
    "MORIGON": "MARIGAON", "MORIGAON": "MARIGAON", "BASKA": "BAKSA", "N C HILLS": "DIMA HASAO",
    "BHANKA": "BANKA", "ZAMUI": "JAMUI", "CHAMPARAN EAST": "PURBI CHAMPARAN",
    "CHAMPARAN WEST": "PASHCHIM CHAMPARAN", "CHAMPARAN E": "PURBI CHAMPARAN",
    "CHAMPARAN W": "PASHCHIM CHAMPARAN",
    "E CHAMPARAN": "PURBI CHAMPARAN", "EAST CHAMPARAN": "PURBI CHAMPARAN",
    "WEST CHAMPARAN": "PASHCHIM CHAMPARAN", "ARAIA": "ARARIA", "BAGHALPUR": "BHAGALPUR",
    "DARBANGA": "DARBHANGA", "MONGHYR": "MUNGER", "MUNGHYR": "MUNGER", "SEKHPURA": "SHEIKHPURA",
    "SUPAUAL": "SUPAUL", "PURNEA": "PURNIA", "BHABUA": "KAIMUR", "BHABHUA": "KAIMUR",
    "MAHADEVPURA": "MADHEPURA",
    "DANTEWARA": "DAKSHIN BASTAR DANTEWADA", "DANTEWADA": "DAKSHIN BASTAR DANTEWADA",
    "KAWARDHA": "KABIRDHAM", "KABIRDHAM": "KABIRDHAM", "KORIYA": "KOREA", "SARGUJA": "SURGUJA",
    "RAJ NANDGAON": "RAJNANDGAON", "JAGDALPUR": "BASTAR", "JASHPUR": "JASHPUR",
    "GARIABAND": "GARIYABAND",
    "GUMAL": "GUMLA", "SINGHBHUMI W": "WEST SINGHBHUM", "SINGHBHUM W": "WEST SINGHBHUM",
    "SINGHBHUM EAST": "EAST SINGHBUM", "DEVDHAR": "DEOGHAR",
    "SARAIKELA": "SARAIKELA KHARSAWAN",
    "BELGAUM": "BELAGAVI", "SHIMOGA": "SHIVAMOGGA", "UTTAR KANADA": "UTTARA KANNADA",
    "UTTARAKANNADA": "UTTARA KANNADA", "UTTAR KANNAD": "UTTARA KANNADA",
    "DAKSHIN KANNADA": "DAKSHINA KANNADA", "DAKSHIN KANNAD": "DAKSHINA KANNADA",
    "UDUPPI": "UDUPI", "YADGIR": "YADAGIRI", "BAGALKOT": "BAGALKOTE", "BELLARY": "BALLARI",
    "BIJAPUR": "VIJAYAPURA", "BANGALORE URBAN": "BENGALURU URBAN", "GULBURGA": "KALABURAGI", "GULBARGA": "KALABURAGI",
    "MYSORE": "MYSURU", "TUMKUR": "TUMAKURU",
    "PALLAKKAD": "PALAKKAD",
    "ANOOPPUR": "ANUPPUR", "SHAHADOL": "SHAHDOL", "CHHATTARPUR": "CHHATARPUR",
    "CHATTARPUR": "CHHATARPUR", "CHINDWARA": "CHHINDWARA", "EAST NIMAR": "KHANDWA",
    "DHAAR": "DHAR", "MANDALA": "MANDLA", "NARSINGPUR": "NARSINGHPUR", "NARASINGHPUR": "NARSINGHPUR",
    "NASIK": "NASHIK", "SHOLAPUR": "SOLAPUR", "JALANA": "JALNA",
    "BOLANGIR": "BALANGIR", "BOUDHA": "BOUDH", "DEOGHAR": "DEOGARH", "NAWAPARA": "NUAPADA",
    "NAWAPADA": "NUAPADA", "NAWRANGPUR": "NABARANGPUR", "NAWARANGPUR": "NABARANGPUR",
    "PHULBANI": "KANDHAMAL", "KEONJHAR": "KENDUJHAR", "ANGUL": "ANUGUL", "JAJPUR": "JAJAPUR",
    "BARAGARH": "BARGARH", "KHURDA": "KHORDHA", "BALASORE": "BALESHWAR",
    "SUBARNAPUR": "SONEPUR", "SONEPUR": "SONEPUR", "SUNDERGARH": "SUNDARGARH",
    "SIVAGANGAI": "SIVAGANGA", "TIRUVANNMALAI": "TIRUVANNAMALAI", "VIRUDUNAGAR": "VIRUDHUNAGAR",
    "THOOTHUKUDI": "TUTICORIN", "NILGIRIS": "THE NILGIRIS",
    "BADAUN": "BUDAUN", "BEHARAICH": "BAHRAICH", "BARELLY": "BAREILLY", "SHIVASTI": "SHRAVASTI",
    "RAEBARELI": "RAE BARELI", "RAI BAREILLY": "RAE BARELI", "SIDDHARTHNAGAR": "SIDDHARTH NAGAR",
    "SIDDHARTHANAGAR": "SIDDHARTH NAGAR",
    "RAVIDAS NAGAR": "SANT RAVIDAS NAGAR", "RAVIDASNAGAR": "SANT RAVIDAS NAGAR",
    "SANTKABIR NAGAR": "SANT KABEER NAGAR", "SANT KABIR NAGAR": "SANT KABEER NAGAR",
    "GAZIPUR": "GHAZIPUR", "CHANDUALI": "CHANDAULI", "CHITRAKUT": "CHITRAKOOT",
    "KHERI": "KHERI", "LAKHIMPUR KHERI": "KHERI", "KUSHINAGAR": "KUSHI NAGAR",
    "FATEPUR": "FATEHPUR", "BALARAMPUR": "BALRAMPUR", "KANPUR CITY": "KANPUR NAGAR",
    "RAEBAREILLY": "RAE BARELI", "MAHAMAYA NAGAR": "HATHRAS", "B SHAHAR": "BULANDSHAHR",
    "BULAND SHAHAR": "BULANDSHAHR", "KANSI RAM NAGAR": "KASGANJ", "SHEOPUR KALAN": "SHEOPUR",
    "UTTARKASHI": "UTTAR KASHI", "RUDRAPRAYAG": "RUDRA PRAYAG", "SINGROLI": "SINGRAULI",
    "CHATRAPATI SHAHU JI MAHARAJ NAGAR": "AMETHI", "GONAD": "GONDA",
    "UDHAMSINGH NAGAR": "UDAM SINGH NAGAR",
    "24 PARGANAS SOUTH": "SOUTH 24 PARGANAS", "MIDNAPUR WEST": "PASHCHIM MEDINIPUR",
    "COOCH BEHAR": "COOCHBEHAR", "DINAJUR NORTH": "UTTAR DINAJPUR", "DINAJPUR NORTH": "UTTAR DINAJPUR",
    "DINAJPUR N": "UTTAR DINAJPUR", "DINAJPUR S": "DAKSHIN DINAJPUR", "PURBI MIDNAPUR": "PURBA MEDINIPUR",
    "MIDNAPUR EAST": "PURBA MEDINIPUR",
    "MIDNAPUR WEST": "PASHCHIM MEDINIPUR", "BIRBHUMI": "BIRBHUM", "MALDA": "MALDAH",
    "BROACH": "BHARUCH", "DOHAD": "DOHAD", "DAHOD": "DOHAD", "KUTCH": "KACHCHH",
    "PANCHMAHALS": "PANCH MAHALS", "PANCHMAHAL": "PANCH MAHALS", "BANASKANTHA": "BANAS KANTHA",
    "SABARKANTHA": "SABAR KANTHA", "MEHSANA": "MAHESANA", "AHMEDABAD": "AHMADABAD",
    "CHHOTA UDEPUR": "CHHOTAUDEPUR",
    "HISSAR": "HISAR", "SONEPAT": "SONIPAT", "JAMUNANAGAR": "YAMUNANAGAR",
    "MAHENDERGARH": "MAHENDRAGARH", "MOHINDERGARH": "MAHENDRAGARH",
    "BHATINDA": "BATHINDA", "FIROZPUR": "FIROZEPUR", "FEROZPUR": "FIROZEPUR",
    "AJITGARH": "S A S NAGAR", "ROOPNAGAR": "RUPNAGAR", "NAWANSHAHR": "SHAHID BHAGAT SINGH NAGAR",
    "N SHAHAR SBS NAGAR": "SHAHID BHAGAT SINGH NAGAR", "CHITTOR": "CHITTORGARH",
    "THOOTHUKKUDI": "TUTICORIN", "SURENDRA NAGAR": "SURENDRANAGAR",
    "BUDGAM": "BADGAM", "SHUPIYAN": "SHOPIAN", "LEH": "LEH LADAKH", "TAMENG LONG": "TAMENGLONG",
    "RIBHOI": "RI BHOI", "JAINTA HILLS": "WEST JAINTIA HILLS", "AIZWAL": "AIZAWL", "L DIBANG VALLEY": "LOWER DIBANG VALLEY", "CHHIMTUIPUI": "SAIHA",
    "ROOPNANAGR": "RUPNAGAR", "ROPAR": "RUPNAGAR", "MOHALI": "S A S NAGAR",
    "TARAN TARAN": "TARN TARAN", "TARANTARAN": "TARN TARAN", "TARN TARAN SAHIB": "TARN TARAN",
    "SANGROOR": "SANGRUR", "HOSIARPUR": "HOSHIARPUR",
    "NAGORE": "NAGAUR", "S MADHOPUR": "SAWAI MADHOPUR",
    "ANANTNAGH": "ANANTNAG", "PULWANNA": "PULWAMA",
    "AIZAWAL": "AIZAWL", "LUGLEI": "LUNGLEI", "LUNGEI": "LUNGLEI",
    "JAINTIA HILLS": "WEST JAINTIA HILLS", "RI BHOI": "RI BHOI",
    "BISHUNPUR": "BISHNUPUR",
}
# Bihar's Aurangabad and Maharashtra's Aurangabad are both legit; the alias table is applied within state.


# aliases that only make sense inside one state
ALIAS_STATE = {
    "SIKKIM": {"EAST SIKKIM": "GANGTOK", "EAST": "GANGTOK", "WEST SIKKIM": "GYALSHING", "WEST": "GYALSHING",
               "SOUTH SIKKIM": "NAMCHI", "NORTH SIKKIM": "MANGAN"},
    "TRIPURA": {"NORTH": "NORTH TRIPURA", "SOUTH": "SOUTH TRIPURA", "WEST": "WEST TRIPURA"},
}


def match_one(state_src, dist_src):
    states = STATE_MAP.get(norm(state_src), [norm(state_src)])
    cand = panel[panel["state_name"].isin(states)]
    if cand.empty:
        return None, None, "state_not_in_panel", np.nan
    d0 = norm(dist_src)
    d = d0
    if not (cand["nd"] == d0).any():
        d = ALIAS_STATE.get(states[0], {}).get(d0, ALIAS.get(d0, d0))
    exact = cand[cand["nd"] == d]
    if len(exact):
        keys = ";".join(exact["district_key"])
        return keys, exact["district_name"].iloc[0], "exact_or_alias", 100.0
    best = process.extractOne(d, cand["nd"].tolist(), scorer=fuzz.token_sort_ratio)
    if best and best[1] >= 85:
        hit = cand[cand["nd"] == best[0]]
        return ";".join(hit["district_key"]), hit["district_name"].iloc[0], "fuzzy", round(best[1], 1)
    return None, None, "unmatched", round(best[1], 1) if best else np.nan


# ================================================================ (1) NFSM lists
def L(s):
    return [x.strip() for x in s.split(",") if x.strip()]


RICE_2007 = {
    "Andhra Pradesh": L("Adilabad, Guntur, Khammam, Krishna, Mahaboobnagar, Medak, Nalgonda, Nellore, Srikakulam, Visakhapatnam, Vizianagarm"),
    "Assam": L("Barpeta, Bongaigaon, Darrang, Dhemaji, Goalpara, Karbi-Anglong, Kokrajhar, Lakhimpur, Morigon, Nalbari, Sonitpur, Nagaon, Tinsukia"),
    "Bihar": L("Araria, Bhanka, Champaran (East), Champaran (West), Darbhanga, Gaya, Katihar, Kishanganj, Madhubani, Madhepura, Muzaffarpur, Nalanda, Saharsa, Samastipur, Sitamarhi, Siwan, Supaul, Zamui"),
    "Chhattisgarh": L("Dantewara, Janjgir-Champa, Jashpur, Kawardha, Korba, Koriya, Raigarh, Raipur, Raj Nandgaon, Sarguja"),
    "Jharkhand": L("Simdega, Singhbhum (W), Ranchi, Gumla, Hazaribagh"),
    "Karnataka": L("Belgaum, Shimoga, Uttarakannada, Dakshin Kannada, Udupi, Hassan, Raichur"),
    "Madhya Pradesh": L("Anuppur, Damoh, Dindori, Katni, Mandla, Panna, Rewa, Satna, Shahdol"),
    "Maharashtra": L("Bhandara, Chandrapur, Gadchiroli, Gondia, Nasik, Pune"),
    "Orissa": L("Bolangir, Jajpur, Dhenkanal, Angul, Kalahandi, Nawapara, Keonjhar, Malkangiri, Nawarangpur, Phulbani, Boudha, Nayagarh, Deoghar, Jharsuguda, Sundargarh"),
    "Tamil Nadu": L("Nagapattinam, Pudukkottai, Ramanathapuram, Sivagangai, Thiruvarur"),
    "Uttar Pradesh": L("Azamgarh, Badaun, Bahraich, Ballia, Balrampur, Banda, Bareilly, Basti, Deoria, Fatehpur, Gonda, Gorakhpur, Ghazipur, Hardoi, Mainpuri, Mau, Mirzapur, Raebareli, Rampur, Saharanpur, Shivasti, Siddharthnagar, Sitapur, Sonbhadra, Sultanpur, Unnao"),
    "West Bengal": L("24 Parganas (South), Cooch-Behar, Dinajur (North), Howrah, Jalpaiguri, Midnapur (East), Midnapur (West), Purulia"),
}
RICE_2009 = {k: list(v) for k, v in RICE_2007.items()}
RICE_2009["Gujarat"] = L("Dahod, Panchmahals")
RICE_2009["Kerala"] = L("Pallakkad")

WHEAT_2007 = {
    "Bihar": L("Araia, Baghalpur, Banka, Bhabua, Champaran (E), Champaran (W), Darbanga, Jamui, Katihar, Khagaria, Kishanganj, Madhubani, Madhepura, Muzaffarpur, Nalanda, Purnia, Rohtas, Samastipur, Saran, Sitamarhi, Supaul, Vaishali, Monghyr, Nawada, Sekhpura"),
    "West Bengal": L("Jalpaiguri, Coochbehar, Dinajpur (N), Dinajpur (S)"),
    "Madhya Pradesh": L("Betul, Bhind, Chattarpur, Damoh, Dewas, East Nimar, Guna, Harda, Indore, Jabalpur, Katni, Raisen, Rajgarh, Rewa, Sagar, Satna, Sehore, Seoni, Shivpuri, Sidhi, Tikamgarh, Ujjain, Vidisha, Balaghat, Dhaar, Dindori, Jhabua, Mandala, Panna, Shahdol"),
    "Punjab": L("Hoshiarpur, Firozpur, Bhatinda, Rupnagar, Amritsar, Sangrur, Gurdaspur"),
    "Haryana": L("Bhiwani, Ambala, Gurgaon, Jamunanagar, Rohtak, Jhajjar, Mahendergarh"),
    "Uttar Pradesh": L("Allahabad, Ambedkar Nagar, Azamgarh, Bahraich, Ballia, Balrampur, Barelly, Basti, Chandauli, Chitrakoot, Deoria, Faizabad, Fatehpur, Gazipur, Gonda, Gorakhpur, Hamirpur, Hardoi, Jaunpur, Jhansi, Kaushambi, Lalitpur, Lucknow, Maharajganj, Mahoba, Mau, Mirzapur, Pratapgarh, Raebareli, Ravidasnagar, Santkabir Nagar, Shravasti, Siddharthnagar, Sitapur, Sultanpur, Unnao, Sonbhadra, Varanasi"),
    "Rajasthan": L("Ajmer, Banswara, Bhilwara, Bikaner, Jaipur, Jalore, Jhalawar, Kota, Nagore, Pali, S. Madhopur, Sikar, Sirohi, Tonk, Udaipur"),
    "Maharashtra": L("Ahmednagar, Aurangabad, Dhule, Nagpur, Nasik, Parbhani, Pune, Solapur"),
    "Gujarat": L("Ahmedabad, Sabarkantha, Banaskantha, Mehsana"),
}
WHEAT_2009 = {k: list(v) for k, v in WHEAT_2007.items()}
WHEAT_2009["Punjab"] = WHEAT_2007["Punjab"] + L("Barnala, Mohali, Taran Taran")
WHEAT_2009["Uttar Pradesh"] = L("Allahabad, Ambedkar Nagar, Azamgarh, Bahraich, Ballia, Balrampur, Barabanki, Bareilly, Basti, Chandauli, Deoria, Faizabad, Fatehpur, Ghazipur, Gonda, Gorakhpur, Hamirpur, Hardoi, Jaunpur, Jhansi, Kaushambi, Kushinagar, Lucknow, Maharajganj, Mainpuri, Mathura, Mau, Pratapgarh, Rai Bareilly, Ravidas Nagar, Sant Kabir Nagar, Shravasti, Siddharthnagar, Sitapur, Sonbhadra, Sultanpur, Unnao, Varanasi")

PULSES_2007 = {
    "Andhra Pradesh": L("Adilabad, Anantpur, Cuddapah, East Godavari, Guntur, Khammam, Krishna, Kurnool, Mahaboobnagar, Nalgonda, Nizamabad, Prakasam, Srikakulam, Warangal"),
    "Bihar": L("Araria, Aurangabad, Bhabhua, Bhojpur, Madhubani, Mahadevpura, Muzaffarpur, Nalanda, Patna, Purnea, Saharsa, Samastipur, Supaul"),
    "Chhattisgarh": L("Bilaspur, Durg, Jashpur, Kawardha, Raigarh, Raipur, Rajnandgaon, Sarguja"),
    "Gujarat": L("Banaskantha, Broach, Dohad, Jamnagar, Kutch, Narmada, Panch Mahals, Patan, Sabarkantha, Surat, Vadodara"),
    "Haryana": L("Bhiwani, Hisar, Rohtak, Sirsa, Sonepat"),
    "Karnataka": L("Bagalkot, Belgaum, Bellary, Bidar, Bijapur, Chitradurga, Dharwad, Gadag, Gulburga, Koppal, Mysore, Raichur, Tumkur"),
    "Madhya Pradesh": L("Chhatarpur, Chindwara, Damoh, Dewas, Guna, Jabalpur, Jhabua, Narsingpur, Panna, Raisen, Rajgarh, Rewa, Sagar, Satna, Seoni, Shajapur, Shivpuri, Tikamgarh, Ujjain, Vidisha"),
    "Maharashtra": L("Ahmednagar, Akola, Amravati, Aurangabad, Buldhana, Chandrapur, Hingoli, Jalgaon, Jalana, Latur, Nagpur, Nanded, Nasik, Osmanabad, Parbhani, Wardha, Washim, Yavatmal"),
    "Orissa": L("Baragarh, Bolangir, Cuttack, Ganjam, Kalahandi, Keonjhar, Khurda, Nayagarh, Puri, Rayagada"),
    "Punjab": L("Amritsar, Ferozpur, Gurdaspur, Ludhiana, Sangrur"),
    "Rajasthan": L("Ajmer, Barmer, Bikaner, Chittorgarh, Churu, Dausa, Ganganagar, Hanumangarh, Jaipur, Jhunjhunu, Jodhpur, Kota, Nagaur, Sikar, Tonk"),
    "Tamil Nadu": L("Cuddalore, Erode, Nagapattinam, Namakkal, Thiruvallur, Thiruvarur, Thoothukudi, Tiruvannmalai, Coimbatore, Vellore, Villupuram, Virudunagar"),
    "Uttar Pradesh": L("Badaun, Beharaich, Ballia, Balrampur, Banda, Barabanki, Chanduali, Chitrakut, Fatehpur, Hamirpur, Jalaun, Jhansi, Kanpur (Dehat), Kaushambi, Kheri, Lalitpur, Mahoba, Mirzapur, Sitapur"),
    "West Bengal": L("Birbhumi, Malda, Murshidabad, Nadia, Purulia"),
}
PULSES_2009 = {k: list(v) for k, v in PULSES_2007.items()}
PULSES_2009["Punjab"] = PULSES_2007["Punjab"] + L("Barnala, Tarantaran")
PULSES_2009["Rajasthan"] = PULSES_2007["Rajasthan"] + L("Pratapgarh")

for dct, tot in [(RICE_2007, 133), (RICE_2009, 136), (WHEAT_2007, 138), (WHEAT_2009, 141),
                 (PULSES_2007, 168), (PULSES_2009, 171)]:
    n = sum(len(v) for v in dct.values())
    assert n == tot, (n, tot)
# per-state counts printed in the 2009 annexure
assert len(RICE_2009["Uttar Pradesh"]) == 26 and len(RICE_2009["Orissa"]) == 15
assert len(WHEAT_2009["Madhya Pradesh"]) == 30 and len(WHEAT_2009["Uttar Pradesh"]) == 38
assert len(PULSES_2009["Madhya Pradesh"]) == 20 and len(PULSES_2009["Maharashtra"]) == 18


def parse_annex(pdf_path, pages):
    """Word-position parse of an NFSM 'Districts covered under' annexure.
    Returns rows (annex_letter, subcrop, state_printed, declared_n, district_printed)."""
    pdf = pdfplumber.open(pdf_path)
    rows, annex, sub, state, decl = [], None, None, None, None
    for pn in pages:
        ws = pdf.pages[pn - 1].extract_words(x_tolerance=2)
        lines = {}
        for w in ws:
            lines.setdefault(round(w["top"] / 3), []).append(w)
        for k in sorted(lines):
            Lw = sorted(lines[k], key=lambda w: w["x0"])
            txt = " ".join(w["text"] for w in Lw)
            m = re.search(r"Annexure-?\s*II\s*\(([a-dA-D])\)", txt)
            if m:
                if m.group(1).lower() != annex:
                    sub = None
                annex = m.group(1).lower()
                continue
            if re.search(r"Rohtak in Haryana|Etah and Ballia|Common districts covered|^Grand Total", txt):
                continue
            if re.search(r"Districts (covered|Covered)|Name of|Name Of|contd|P a g e|^S\.|^State", txt) \
                    or re.fullmatch(r"\d+", txt):
                continue
            if txt.strip() in ("MAIZE", "BARLEY"):
                sub = txt.strip().lower(); continue
            # state header: optional serial number, words, '(n)'
            j = next((i for i, w in enumerate(Lw) if re.search(r"\(\d+\)$", w["text"]) and w["x0"] < 260), None)
            if j is not None:
                st_words = [w["text"] for w in Lw[:j + 1]]
                if re.fullmatch(r"\d+\.?", st_words[0]):
                    st_words = st_words[1:]
                st = " ".join(st_words)
                mm = re.search(r"^(.*?)\s*\((\d+)\)$", st)
                state, decl = mm.group(1).strip(), int(mm.group(2))
                rest = Lw[j + 1:]
            else:
                rest = Lw
            if not rest:
                continue
            cells, cur = [], [rest[0]]
            for a, b in zip(rest, rest[1:]):
                if b["x0"] - a["x1"] > 10:
                    cells.append(cur); cur = [b]
                else:
                    cur.append(b)
            cells.append(cur)
            for c in cells:
                t = " ".join(w["text"] for w in c).strip().rstrip("*").strip()
                if t.lower() == "barley":
                    sub = "barley"; continue
                if t.lower() == "maize":
                    sub = "maize"; continue
                if t in ("-", "") or state is None or re.match(r"^(Total|Grand|Contd)", t):
                    continue
                rows.append([annex, sub, state, decl, t])
    # repair names wrapped onto a second line: an open parenthesis is closed by a later cell
    # of the same state; Meghalaya prints 'District, HQ town' and wraps the town
    fixed, pend = [], {}
    for r in rows:
        key = (r[0], r[1], r[2])
        t = r[4]
        if key in pend and (t.endswith(")") and "(" not in t):
            fixed[pend.pop(key)][4] += " " + t
            continue
        if r[2] == "Meghalaya" and "Hills" not in t and "District" not in t:
            continue      # wrapped headquarters town, not a district
        if t.count("(") > t.count(")"):
            pend[key] = len(fixed)
        if r[2] == "Meghalaya" and "," in t:
            r[4] = t.split(",")[0].strip()
        fixed.append(r)
    return [tuple(r) for r in fixed]


def check_counts(rows, label):
    df = pd.DataFrame(rows, columns=["annex", "sub", "state", "decl", "district"])
    bad = []
    for (a, s, st), g in df.groupby(["annex", "sub", "state"], dropna=False):
        if len(g) != g["decl"].iloc[0]:
            bad.append((label, a, s, st, len(g), g["decl"].iloc[0]))
    return df, bad


xii_rows = parse_annex(RAW + "/nfsm_NFSMXII.pdf", range(42, 47))          # rice, wheat
xii_cc = parse_annex(RAW + "/nfsm_NFSMXII.pdf", range(53, 56))            # coarse cereals
y18_rows = parse_annex(RAW + "/nfsm_Guideline_nfsmandoilseed201819to201920.pdf", range(63, 68))
y18_cc = parse_annex(RAW + "/nfsm_Guideline_nfsmandoilseed201819to201920.pdf", range(78, 83))

parsed = {}
all_bad = []
for lab, rows in [("xii", xii_rows + xii_cc), ("y18", y18_rows + y18_cc)]:
    df, bad = check_counts(rows, lab)
    parsed[lab] = df
    all_bad += bad
print("NFSM parsed lists; state-count mismatches:", all_bad if all_bad else "none")
assert not all_bad, all_bad
COMP = {"a": "rice", "b": "wheat", "d": "coarse_cereals"}

# pulses in XII and 2018: coverage is all districts of listed states; take states and counts
def pulses_states(pdf_path, pages):
    pdf = pdfplumber.open(pdf_path)
    txt = "\n".join(pdf.pages[p - 1].extract_text() or "" for p in pages)
    out = []
    for m in re.finditer(r"([A-Z][A-Za-z&. ]+?)\s*\((\d+)\)", txt):
        nm = re.sub(r"^\d+\.?\s*", "", m.group(1)).strip()
        nm = re.sub(r"^(Districts covered under NFSM-Pulses|S\. No\. Name of State)\s*", "", nm)
        out.append((nm, int(m.group(2))))
    return out


xii_p = [s for s in pulses_states(RAW + "/nfsm_NFSMXII.pdf", range(47, 53)) if s[0] not in ("NFSM-Pulses",)]
y18_p = pulses_states(RAW + "/nfsm_Guideline_nfsmandoilseed201819to201920.pdf", range(68, 78))
xii_p = [(n, k) for n, k in xii_p if n != "Districts covered under NFSM-Pulses" and k != 557]
y18_p = [(n, k) for n, k in y18_p if k != 638]
print("XII pulses state headers:", xii_p)
print("2018 pulses state headers:", y18_p)

# Nutri-cereals 2018-19 (tables)
nut = []
pdf18 = pdfplumber.open(RAW + "/nfsm_Guideline_nfsmandoilseed201819to201920.pdf")
st = None
for pn in range(138, 143):
    for t in pdf18.pages[pn - 1].extract_tables():
        for r in t:
            if not r or len(r) < 7 or r[2] is None:
                continue
            if r[1] and re.search(r"\(\d+\)", r[1]):
                st = r[1]
            d = (r[2] or "").replace("\n", " ").strip()
            if not d or re.search(r"State.?Total|Name of Districts|Grand", d, re.I) or st is None:
                continue
            flags = [(c or "").strip().lower().startswith("covered") for c in r[3:7]]
            nut.append((st, d, *flags))
nut = pd.DataFrame(nut, columns=["state_printed", "district_printed", "jowar", "bajra", "ragi", "small_millets"])
nut["state"] = nut["state_printed"].str.replace(r"\s*\(\d+\)", "", regex=True).str.strip()
nut["decl"] = nut["state_printed"].str.extract(r"\((\d+)\)").astype(float)
cnt = nut.groupby("state_printed").size()
nb = [(s, n) for s, n in cnt.items() if n != nut.loc[nut.state_printed == s, "decl"].iloc[0]]
print("Nutri-cereal states:", len(cnt), "districts:", len(nut), "count mismatches:", nb)

# ---------------------------------------------------------------- assemble list rows
recs = []


def add(component, vintage, src, dct=None, df=None, sub=None):
    if dct is not None:
        for s, ds in dct.items():
            for d in ds:
                recs.append((component, vintage, sub, src, s, d))
    if df is not None:
        for _, r in df.iterrows():
            recs.append((component, vintage, r.get("sub"), src, r["state"], r["district"]))


add("rice", "2007", "nfsm_2007", RICE_2007)
add("rice", "2009", "nfsm_2009", RICE_2009)
add("wheat", "2007", "nfsm_2007", WHEAT_2007)
add("wheat", "2009", "nfsm_2009", WHEAT_2009)
add("pulses", "2007", "nfsm_2007", PULSES_2007)
add("pulses", "2009", "nfsm_2009", PULSES_2009)
for lab, src, vint in [("xii", "nfsm_xii", "2014"), ("y18", "nfsm_2018", "2018")]:
    df = parsed[lab]
    for a, comp in COMP.items():
        add(comp, vint, src, df=df[df["annex"] == a])
for _, r in nut.iterrows():
    crops = [c for c in ["jowar", "bajra", "ragi", "small_millets"] if r[c]]
    recs.append(("nutri_cereals", "2018", "+".join(crops) if crops else None, "nfsm_2018",
                 r["state"], r["district_printed"]))

lists = pd.DataFrame(recs, columns=["component", "vintage", "sub_component", "source_id",
                                    "state_printed", "district_printed"])
m = lists.apply(lambda r: match_one(r["state_printed"], r["district_printed"]), axis=1, result_type="expand")
lists[["panel_district_key", "panel_district_name", "match_method", "match_score"]] = m
lists["source_doc"] = lists["source_id"].map(lambda s: SOURCES[s][0])
lists["source_url"] = lists["source_id"].map(lambda s: SOURCES[s][1])
lists.to_csv(OUT + "/policy_nfsm_district_lists.csv", index=False)
un = lists[lists["match_method"].isin(["unmatched", "state_not_in_panel"])]
un.to_csv(OUT + "/policy_nfsm_unmatched.csv", index=False)
print("\nNFSM list rows:", len(lists))
print(lists.groupby(["component", "vintage"]).size().to_string())
print("match methods:\n", lists["match_method"].value_counts().to_string())
print("unmatched:", len(un))
if len(un):
    print(un[["component", "vintage", "state_printed", "district_printed", "match_score"]].to_string())

# ---------------------------------------------------------------- state rules
ISOPOM14 = ["ANDHRA PRADESH", "TELANGANA", "BIHAR", "CHHATTISGARH", "GUJARAT", "HARYANA", "KARNATAKA",
            "MADHYA PRADESH", "MAHARASHTRA", "ODISHA", "PUNJAB", "RAJASTHAN", "TAMIL NADU",
            "UTTAR PRADESH", "WEST BENGAL"]          # 14 states; Telangana is part of pre-2014 AP


def panel_states(names):
    out = []
    for n in names:
        out += STATE_MAP.get(norm(n), [norm(n)])
    return sorted(set(out))


xii_pulse_states = []
for n, k in xii_p:
    if "," in n or "Arunachal" in n:
        continue
    xii_pulse_states.append(n)
NE8 = ["ARUNACHAL PRADESH", "ASSAM", "MANIPUR", "MEGHALAYA", "MIZORAM", "NAGALAND", "SIKKIM", "TRIPURA"]
rules = []
for y in ["2010-11", "2011-12", "2012-13", "2013-14"]:
    conf = "medium" if y in ("2010-11", "2011-12") else "low"
    for s in ISOPOM14:
        rules.append(("pulses", y, s, "all_districts", conf, "nfsm_csd"))
    for s in ["ASSAM", "JHARKHAND"]:
        rules.append(("pulses", y, s, "some_districts_unnamed", conf, "nfsm_2010_expansion"))
xii_pulse_states = [s for s in panel_states(xii_pulse_states) if s in set(panel.state_name)]
for s in sorted(set(xii_pulse_states) | set(NE8)):
    rules.append(("pulses", "2014-15", s, "all_districts", "high", "nfsm_xii"))
y18_states = panel_states([n for n, k in y18_p])
for y in ["2015-16", "2016-17", "2017-18", "2018-19", "2019-20"]:
    conf = "high" if y in ("2018-19", "2019-20") else "medium"
    for s in y18_states:
        rules.append(("pulses", y, s, "all_districts", conf, "nfsm_2018"))
rules = pd.DataFrame(rules, columns=["component", "year", "panel_state", "coverage", "confidence", "source_id"])
rules["source_url"] = rules["source_id"].map(lambda s: SOURCES[s][1])
rules.to_csv(OUT + "/policy_nfsm_state_rules.csv", index=False)
print("\n2018 pulses states ({}): {}".format(len(y18_states), y18_states))
print("XII pulses states ({}): {}".format(len(xii_pulse_states), xii_pulse_states))

# ---------------------------------------------------------------- district x year
# vintage in force by year; confidence reflects how directly a document ties list to year
VINT = {
    "rice": {"2007-08": ("2007", "high"), "2008-09": ("2007", "high"), "2009-10": ("2009", "high"),
             "2010-11": ("2009", "medium"), "2011-12": ("2009", "medium"),
             "2012-13": ("2009", "low"), "2013-14": ("2009", "low"),
             "2014-15": ("2014", "high"), "2015-16": ("2018", "medium"), "2016-17": ("2018", "medium"),
             "2017-18": ("2018", "medium"), "2018-19": ("2018", "high"), "2019-20": ("2018", "high")},
}
VINT["wheat"] = VINT["rice"]
VINT["pulses"] = {"2007-08": ("2007", "high"), "2008-09": ("2007", "high"), "2009-10": ("2009", "high")}
VINT["coarse_cereals"] = {"2014-15": ("2014", "high"), "2015-16": ("2014", "low"), "2016-17": ("2014", "low"),
                          "2017-18": ("2014", "low"), "2018-19": ("2018", "high"), "2019-20": ("2018", "high")}
VINT["nutri_cereals"] = {"2018-19": ("2018", "high"), "2019-20": ("2018", "high")}


def keyset(comp, vint):
    s = lists[(lists.component == comp) & (lists.vintage == vint)]["panel_district_key"].dropna()
    out = set()
    for k in s:
        out |= set(k.split(";"))
    return out


grid = []
for comp in ["rice", "wheat", "pulses", "coarse_cereals", "nutri_cereals"]:
    for y in YEARS:
        base = pd.DataFrame({"district_key": panel["district_key"], "state_name": panel["state_name"]})
        base["component"], base["year"] = comp, y
        base["treated"], base["basis"], base["confidence"], base["source_id"] = 0, "not_listed", None, None
        if comp in VINT and y in VINT[comp]:
            v, conf = VINT[comp][y]
            ks = keyset(comp, v)
            base.loc[base.district_key.isin(ks), "treated"] = 1
            base.loc[base.district_key.isin(ks), "basis"] = "district_list"
            base["confidence"] = conf
            base["source_id"] = {"2007": "nfsm_2007", "2009": "nfsm_2009", "2014": "nfsm_xii", "2018": "nfsm_2018"}[v]
        elif comp == "pulses":
            r = rules[(rules.component == "pulses") & (rules.year == y)]
            full = set(r.loc[r.coverage == "all_districts", "panel_state"])
            part = set(r.loc[r.coverage == "some_districts_unnamed", "panel_state"])
            base.loc[base.state_name.isin(full), ["treated", "basis"]] = [1, "all_districts_of_state"]
            base.loc[base.state_name.isin(part), ["treated", "basis"]] = [np.nan, "state_partly_covered_list_not_found"]
            base["confidence"] = r["confidence"].iloc[0] if len(r) else None
            base["source_id"] = r["source_id"].iloc[0] if len(r) else None
        elif comp in ("coarse_cereals", "nutri_cereals"):
            base["basis"] = "component_not_running"
        grid.append(base)
grid = pd.concat(grid, ignore_index=True)
grid["in_606"] = grid["district_key"].isin(S606)
# rice 2010-11 onward: two extra Jharkhand districts were added (138 vs 136) and are not named
jh = (grid.component == "rice") & grid.year.isin(["2010-11", "2011-12", "2012-13", "2013-14"]) & \
     (grid.state_name == "JHARKHAND") & (grid.treated == 0)
grid.loc[jh, "basis"] = "not_listed_but_2_unnamed_JH_districts_added"
grid.to_csv(OUT + "/policy_nfsm_district_year.csv", index=False)
summ = grid[grid.in_606].groupby(["component", "year"])["treated"].agg(["sum", "count"])
print("\nNFSM treated districts within the 606 sample:\n", summ.to_string())

# ================================================================ (2) Odisha Millets Mission
# 2017-18 block names: OMM AR 2017-18 Annexure 2 (29 of 30 named; the table prints 3 for Rayagada
# while every other OMM table gives Rayagada 4 blocks in 2017-18).
# District entries by year: OMM AR 2020-21 phase-wise scale-up table.
# Block names for 2018-19 onward: 84 'existing' blocks listed in the 2022-26 guideline, minus the
# 2017-18 blocks, assigned to the district's entry year where the district entered in one year.
PHASE = {  # district: {year: n_blocks added}  (OMM AR 2020-21)
    "Bargarh": {"2019-20": 5}, "Bolangir": {"2018-19": 4}, "Gajapati": {"2017-18": 4},
    "Ganjam": {"2018-19": 4}, "Kalahandi": {"2017-18": 4}, "Kandhamal": {"2017-18": 4, "2018-19": 3},
    "Keonjhar": {"2019-20": 7}, "Koraput": {"2017-18": 7, "2018-19": 3}, "Malkangiri": {"2017-18": 4},
    "Mayurbhanj": {"2018-19": 3}, "Nabarangpur": {"2019-20": 5}, "Nuapada": {"2017-18": 3},
    "Rayagada": {"2017-18": 4, "2018-19": 4}, "Sundargarh": {"2018-19": 4, "2020-21": 4},
    "Angul": {"2021-22": 5},   # 84 blocks in 15 districts in 2021-22 = 76 + Angul 5 + Sundargarh 3
}
PHASE["Sundargarh"]["2021-22"] = 3
B1718 = {"Malkangiri": "K.Gumma, Korkunda, Khairput, Mathili",
         "Kandhamal": "Phiringia, Raikia, Kotagarh, Daringibadi",
         "Koraput": "Semiliguda, Dasamantpur, Lamtaput, Boipariguda, Kundra, Borigumma, Nandapur",
         "Kalahandi": "Th.Rampur, Lanjigarh, Narla, Bhawanipatna",
         "Gajapati": "Mohana, Gumma, Rayagada, R.Udayagiri",
         "Rayagada": "Rayagada, Gudari, Gunupur",
         "Nuapada": "Sinapali, Bhoden, Komna"}
EXIST84 = {  # OMM guideline 2022-26, 'Name of Existing Blocks'
    "Angul": "Angul, Athamallick, Chendipada, Kishornagar, Pallahara",
    "Bargarh": "Padampur, Bijepur, Gaisilet, Paikamal, Jharabandha",
    "Bolangir": "Tureikela, Bangamunda, Muribahal, Khaprakhol",
    "Gajapati": "Gumma, Mohana, Rayagada, R. Udayagiri",
    "Ganjam": "Polsara, Patrapur, Dharakot, Sorada",
    "Kalahandi": "Lanjigarh, Thuamula Rampur, Bhavanipatna, Narla",
    "Kandhamal": "Phiringia, Daringbadi, Raikia, Kotagarh, Tumudibandh, Baliguda, K Nuagaon",
    "Keonjhar": "Bansapala, Champua, Harichandanpur, Hatadihi, Joda, Jhumpura, Sadar",
    "Koraput": "Lamptaput, Nandapur, Dasmantapur, Boipariguda, Kundra, Borigumma, Semiliguda, Laxmipur, Koraput, Pottangi",
    "Malkangiri": "Korkonda, K Gumma, Mathili, Khairput",
    "Mayurbhanj": "Jashipur, Bisoi, Bangiriposi",
    "Nabarangpur": "Nabarangapur, Papadahandi, Kosagumuda, Umerkote, Jharigaon",
    "Nuapada": "Boden, Komana, Sinapalli",
    "Rayagada": "Gunpur, Chandrapur, Gudari, Rayagada, Muniguda, Bissamcuttack, Kalyansinghpur, Kashipur",
    "Sundargarh": "Kutra, Rajgangpur, Nuagaon, Kuanrmunda, Balisankara, Gurundia, Hemgiri, Lahunipada, Lathikata, Lephripada, Tangarpalli",
}
NEW58 = {  # 2022-23 additions, same guideline
    "Bargarh": "Bhatli, Sohela", "Bolangir": "Belpada, Titlagarh, Pattanagarh",
    "Boudh": "Harbhanga, Kantamal, Boudh",
    "Dhenkanal": "Odapada, Hindol, Bhuban, Kankadahad, Kamakshyanagar",
    "Gajapati": "Nuagada, Kashinagar", "Ganjam": "Sanakhemundi, Sheragada, Jagannath Prasad",
    "Jharsuguda": "Kolabira, Laikera, Kirmira, Lakhanpur, Jharsuguda",
    "Kalahandi": "Golamunda, M.Rampur, Koksara, Karlamunda",
    "Kandhamal": "Tikabali, Khajuripada, G.Udayagiri, Chakapada",
    "Keonjhar": "Patna, Telkoi, Saharpada, Ghatagaon",
    "Koraput": "Narayanpatna, Bandhugaon", "Malkangiri": "Malkangiri",
    "Mayurbhanj": "Thakurmunda, Kusumi, Morada, Rairangpur, Karanjia, Raruan, Tirring, Bijatala, Suliapada",
    "Nabarangpur": "Tentulikhunti, Nandahandi, Dabugaon, Raighar",
    "Nayagarh": "Dasapalla, Gania, Nuagaon, Odagaon", "Nuapada": "Nuapada, Khariar",
    "Rayagada": "Kolnara",
}
NEW34 = {  # later modification, Annexure-I (expansion 143 -> 177); year of approval not printed clearly
    "Bargarh": "Ambabhona", "Mayurbhanj": "Bahalda, Jamda, Gopabandhu Nagar, Sukruli",
    "Nabarangpur": "Chandahandi", "Nayagarh": "Bhapur", "Rayagada": "Ramanguda",
    "Sundargarh": "Subdega, Sundargarh Sadar, Bargaon", "Bhadrak": "Bonth", "Balasore": "Nilgiri, Oupada",
    "Cuttack": "Tigiria, Badamba, Narsinghpur", "Deogarh": "Teleibani", "Jagatsinghpur": "Biridi",
    "Jajpur": "Sukinda, Danagadi", "Khurda": "Tangi, Banpur", "Kendrapara": "Kendrapara, Derabis",
    "Puri": "Krushnaprasad", "Sambalpur": "Kuchinda, Bamra, Jamankira, Naktideula, Jujumara, Rairakhol",
    "Subarnapur": "Ulunda, Birmaharajpur",
}
assert sum(len(L(v)) for v in EXIST84.values()) == 84
assert sum(len(L(v)) for v in NEW58.values()) == 58
assert sum(len(L(v)) for v in B1718.values()) == 29
assert sum(sum(v.values()) for v in PHASE.values()) == 84


def bkey(b):
    b = norm(b).replace(" ", "")
    return {"THRAMPUR": "THUAMULARAMPUR", "THUAMULARAMPUR": "THUAMULARAMPUR", "BHAWANIPATNA": "BHAVANIPATNA",
            "DARINGIBADI": "DARINGBADI", "KORKUNDA": "KORKONDA", "DASAMANTPUR": "DASMANTAPUR",
            "LAMTAPUT": "LAMPTAPUT", "BHODEN": "BODEN", "KOMNA": "KOMANA", "SINAPALI": "SINAPALLI",
            "GUNUPUR": "GUNPUR", "RUDAYAGIRI": "RUDAYAGIRI", "KGUMMA": "KGUMMA"}.get(b, b)


ob = []
for dist, blocks in EXIST84.items():
    first = B1718.get(dist, "")
    firstk = {bkey(x) for x in L(first)}
    years = sorted(PHASE[dist])
    for b in L(blocks):
        if bkey(b) in firstk:
            ob.append((dist, b, "2017-18", "named in 2017-18 first-phase list", "high", "omm_ar_1718"))
        elif len(years) == 1:
            ob.append((dist, b, years[0], "district entered in a single year", "high", "omm_guideline_2226"))
        else:
            later = [y for y in years if y != "2017-18"]
            if dist == "Rayagada":
                ob.append((dist, b, "2017-18 or 2018-19",
                           "Rayagada had 4 blocks in 2017-18 but only 3 are named; this block is either the "
                           "unnamed fourth (2017-18) or one of the 4 added in 2018-19", "medium", "omm_guideline_2226"))
            elif len(later) == 1:
                ob.append((dist, b, later[0], "not in 2017-18 list; district's only later entry year",
                           "high", "omm_guideline_2226"))
            else:
                ob.append((dist, b, " or ".join(later), "district added blocks in several years; "
                           "block-level year not published", "low", "omm_guideline_2226"))
for dist, blocks in NEW58.items():
    for b in L(blocks):
        ob.append((dist, b, "2022-23", "new block in 2022-26 guideline", "high", "omm_guideline_2226"))
for dist, blocks in NEW34.items():
    for b in L(blocks):
        ob.append((dist, b, "2023-24 (approx.)", "expansion 143 to 177 blocks; approval year approximate",
                   "low", "omm_modifications"))
ob = pd.DataFrame(ob, columns=["district_printed", "block", "entry_year", "basis", "confidence", "source_id"])
ob["source_url"] = ob["source_id"].map(lambda s: SOURCES[s][1])
m = ob.apply(lambda r: match_one("Odisha", r["district_printed"]), axis=1, result_type="expand")
ob[["panel_district_key", "panel_district_name", "match_method", "match_score"]] = m
ob.to_csv(OUT + "/policy_omm_blocks.csv", index=False)
print("\nOMM blocks:", len(ob), "| unmatched districts:",
      ob.loc[ob.match_method == "unmatched", "district_printed"].unique().tolist())

od = panel[panel.state_name == "ODISHA"][["district_key", "district_name", "in_606"]]
rows = []
dmap = ob.drop_duplicates("district_printed").set_index("district_printed")["panel_district_key"].to_dict()
for _, r in od.iterrows():
    dists = [d for d, k in dmap.items() if k == r["district_key"]]
    ph = PHASE.get(dists[0], {}) if dists else {}
    cum = 0
    for y in [f"{a}-{str(a + 1)[-2:]}" for a in range(2014, 2022)]:
        cum += ph.get(y, 0)
        rows.append((r["district_key"], r["district_name"], r["in_606"], y, cum, int(cum > 0),
                     min([k for k in ph], default=None)))
odx = pd.DataFrame(rows, columns=["district_key", "district_name", "in_606", "year", "omm_blocks_cum",
                                  "omm_any", "omm_entry_year"])
odx["source_url"] = SOURCES["omm_ar_2021"][1]
odx.to_csv(OUT + "/policy_omm_district_year.csv", index=False)
print(odx[odx.year == "2019-20"].query("omm_any==1")[["district_name", "omm_blocks_cum", "omm_entry_year"]].to_string())

# ================================================================ (3) MSP
def words_table(pdf_path, page, x_tol=1.5):
    p = pdfplumber.open(pdf_path).pages[page - 1]
    ws = p.extract_words(x_tolerance=x_tol)
    lines = {}
    for w in ws:
        lines.setdefault(round(w["top"] / 2.5), []).append(w)
    return [sorted(lines[k], key=lambda w: w["x0"]) for k in sorted(lines)]


def num(t):
    t = re.sub(r"[^0-9.]", "", t)
    return float(t) if t and re.fullmatch(r"\d+(\.\d+)?", t) else None


def grab(lines, label_regex, cols, tol=14, start_after=None, nth=0):
    """Find the nth line whose leading words match label_regex; assign numeric tokens to the
    nearest column centre (x of the column header right edge)."""
    hits = []
    for Lw in lines:
        lab = " ".join(w["text"] for w in Lw if w["x0"] < cols[0][1] - 25)
        if re.search(label_regex, lab, re.I):
            hits.append(Lw)
    if len(hits) <= nth:
        return {}
    Lw = hits[nth]
    out = {}
    for w in Lw:
        v = num(w["text"])
        if v is None or w["x0"] < cols[0][1] - 25:
            continue
        xc = w["x1"]
        best = min(cols, key=lambda c: abs(c[1] - xc))
        if abs(best[1] - xc) <= tol and best[0] not in out:   # first token wins; later ones are footnote marks
            out[best[0]] = (v, w["text"])
    return out


msp = []   # crop, variety, crop_year, value, raw_token, source_id


def put(crop, variety, d, src):
    for y, (v, raw) in d.items():
        msp.append((crop, variety, y, v, raw, src))


# --- ES 1999-2000: 1997-98, 1998-99, 1999-2000 (right edges of the value columns)
es99 = words_table(RAW + "/es1999_2000_app55.pdf", 1)
C99 = [("1997-98", 447), ("1998-99", 492), ("1999-00", 537)]
for lab, crop, var in [(r"^Common", "paddy", "common"), (r"^Grade", "paddy", "grade_a"),
                       (r"^Wheat", "wheat", ""), (r"^Coarse", "jowar", ""), (r"^Coarse", "bajra", ""),
                       (r"^Coarse", "ragi", ""), (r"^Maize", "maize", ""), (r"^Barley", "barley", ""),
                       (r"^Gram", "gram", ""), (r"^Arhar", "arhar", ""), (r"^Moong", "moong", ""),
                       (r"^Urad", "urad", ""), (r"^Sugarcane", "sugarcane", ""),
                       (r"^F-414", "cotton", "medium_staple"), (r"^H-4", "cotton", "long_staple"),
                       (r"^Groundnut", "groundnut", ""), (r"^Jute", "jute", ""),
                       (r"^Rapeseed", "rapeseed_mustard", ""), (r"^Sunflower", "sunflower", ""),
                       (r"^Black", "soybean", "black"), (r"^Yellow", "soybean", "yellow"),
                       (r"^Safflower", "safflower", ""), (r"^Toria", "toria", ""),
                       (r"^Sesamum", "sesamum", ""), (r"^Niger", "nigerseed", "")]:
    put(crop, var, grab(es99, lab, C99, tol=16), "es_9900")

# --- DES 2004: 2000-01 .. 2004-05 (skip 1998-99/1999-2000 already from ES; 2002-03 SDR column skipped)
d04 = words_table(RAW + "/des_msp_2004.pdf", 1) + words_table(RAW + "/des_msp_2004.pdf", 2)
C04 = [("1999-00", 406), ("2000-01", 460), ("2001-02", 510), ("2002-03", 548), ("2003-04", 633), ("2004-05", 680)]
for lab, crop, var in [(r"PADDY", "paddy", "common"), (r"^Grade", "paddy", "grade_a"),
                       (r"JOWAR", "jowar", ""), (r"BAJRA", "bajra", ""), (r"MAIZE", "maize", ""),
                       (r"RAGI", "ragi", ""), (r"WHEAT", "wheat", ""), (r"BARLEY", "barley", ""),
                       (r"GRAM", "gram", ""), (r"ARHAR", "arhar", ""), (r"MOONG", "moong", ""),
                       (r"URAD", "urad", ""), (r"MASUR", "masur", ""), (r"SUGARCANE", "sugarcane", ""),
                       (r"COTTON", "cotton", "medium_staple"), (r"^H-4", "cotton", "long_staple"),
                       (r"GROUNDNUT", "groundnut", ""), (r"JUTE", "jute", ""),
                       (r"RAPESEED", "rapeseed_mustard", ""), (r"SUNFLOWER", "sunflower", ""),
                       (r"SOYABEEN", "soybean", "black"), (r"^YELLOW", "soybean", "yellow"),
                       (r"SAFFLOWER", "safflower", ""), (r"TORIA", "toria", ""),
                       (r"SESAMUM", "sesamum", ""), (r"NIGERSEED", "nigerseed", "")]:
    put(crop, var, grab(d04, lab, C04, tol=18), "des_2004")

# --- ES 2012-13: 2005-06 .. 2012-13
es13 = words_table(RAW + "/es_es2012-13_estat1.pdf", 74)
C13 = [("2004-05", 272), ("2005-06", 304), ("2006-07", 340), ("2007-08", 370), ("2008-09", 400), ("2009-10", 430),
       ("2010-11", 460), ("2011-12", 490), ("2012-13", 520)]
for lab, crop, var, nth in [(r"Paddy \(Common", "paddy", "common", 0), (r"Paddy \(Grade", "paddy", "grade_a", 0),
                            (r"^Wheat", "wheat", "", 0), (r"Jowar \(Hybrid", "jowar", "", 0),
                            (r"Jowar \(Maldandi", "jowar", "maldandi", 0), (r"^Bajra", "bajra", "", 0),
                            (r"^Ragi", "ragi", "", 0), (r"^Maize", "maize", "", 0), (r"^Barley", "barley", "", 0),
                            (r"^Gram", "gram", "", 0), (r"^Masur", "masur", "", 0), (r"^Arhar", "arhar", "", 0),
                            (r"^Moong", "moong", "", 0), (r"^Urad", "urad", "", 0),
                            (r"Statutory", "sugarcane", "", 0), (r"F-414", "cotton", "medium_staple", 0),
                            (r"Cotton H-4", "cotton", "long_staple", 0), (r"^Groundnut", "groundnut", "", 0),
                            (r"^Jute", "jute", "", 0), (r"Rapeseed", "rapeseed_mustard", "", 0),
                            (r"^Sunflower", "sunflower", "", 0), (r"\(Black", "soybean", "black", 0),
                            (r"\(Yellow", "soybean", "yellow", 0), (r"^Safflower", "safflower", "", 0),
                            (r"^Toria", "toria", "", 0), (r"^Sesamum", "sesamum", "", 0),
                            (r"^Niger", "nigerseed", "", 0)]:
    put(crop, var, grab(es13, lab, C13, tol=16, nth=nth), "es_1213")
# values the ES 2012-13 table sets on an unlabelled line beneath the row (wheat bonus years,
# moong, long-staple cotton); read off the word-position dump and entered here
for crop, var, y, v, raw in [("wheat", "", "2005-06", 650, "650g"), ("wheat", "", "2006-07", 750, "750h"),
                             ("moong", "", "2007-08", 1700, "1700d"), ("moong", "", "2010-11", 3170, "3170l"),
                             ("moong", "", "2011-12", 3500, "3500l"), ("sugarcane", "", "2011-12", 145, "145.00m"),
                             ("cotton", "long_staple", "2006-07", 1990, "1990f"),
                             ("cotton", "long_staple", "2007-08", 2030, "2030f"),
                             ("cotton", "long_staple", "2008-09", 3000, "3000j"),
                             ("cotton", "long_staple", "2009-10", 3000, "3000j"),
                             ("cotton", "long_staple", "2010-11", 3000, "3000j"),
                             ("cotton", "long_staple", "2011-12", 3300, "3300j")]:
    msp.append((crop, var, y, v, raw, "es_1213"))

# --- CACP 2012-13 .. 2021-22: 'Fixed' column of each year
cacp = words_table(RAW + "/msp_since_2012_13.pdf", 1)
CF = [("2012-13", 160), ("2013-14", 230), ("2014-15", 292), ("2015-16", 362), ("2016-17", 430),
      ("2017-18", 498), ("2018-19", 566), ("2019-20", 634), ("2020-21", 700)]
for lab, crop, var in [(r"Paddy Common", "paddy", "common"), (r"Grade", "paddy", "grade_a"),
                       (r"Jowar-Hybrid", "jowar", ""), (r"Jowar-Maldandi", "jowar", "maldandi"),
                       (r"^Bajra", "bajra", ""), (r"^Maize", "maize", ""), (r"^Ragi", "ragi", ""),
                       (r"Tur", "arhar", ""), (r"^Moong", "moong", ""), (r"^Urad", "urad", ""),
                       (r"^Groundnut", "groundnut", ""), (r"Sunflower", "sunflower", ""),
                       (r"Soyabean Black", "soybean", "black"), (r"Soyabean Yellow", "soybean", "yellow"),
                       (r"^Sesamum", "sesamum", ""), (r"^Nigerseed", "nigerseed", ""),
                       (r"Medium Staple", "cotton", "medium_staple"), (r"^Wheat", "wheat", ""),
                       (r"^Barley", "barley", ""), (r"^Gram", "gram", ""), (r"Lentil", "masur", ""),
                       (r"Rapeseed", "rapeseed_mustard", ""), (r"^Safflower", "safflower", ""),
                       (r"^Jute", "jute", ""), (r"^Sugarcane", "sugarcane", "")]:
    put(crop, var, grab(cacp, lab, CF, tol=20), "cacp_1221")
# long-staple cotton sits on the line after 'Medium Staple Cotton' with a '^' label
put("cotton", "long_staple", grab(cacp, r"^Long Staple", CF, tol=20), "cacp_1221")

# ES 1999-2000 prints wheat 1997-98 as '510' with footnote 6 glued on ('5106'): Rs 455 MSP + Rs 55 bonus
msp = [r if not (r[0] == "wheat" and r[2] == "1997-98") else ("wheat", "", "1997-98", 510.0, "510(6)", r[5])
       for r in msp]
# ES 1999-2000 prints medium-staple cotton 1998-99 as '1440' with footnote 7 glued on ('14407'), on its own line
msp.append(("cotton", "medium_staple", "1998-99", 1440.0, "1440(7)", "es_9900"))
msp = pd.DataFrame(msp, columns=["crop", "variety", "crop_year", "msp_rs_per_qtl", "raw_token", "source_id"])
# precedence when two sources give the same year: later compilation wins for its own span
PREC = {"es_9900": 1, "des_2004": 2, "es_1213": 3, "cacp_1221": 4}
SPAN = {"es_9900": ("1997-98", "1999-00"), "des_2004": ("1999-00", "2004-05"),
        "es_1213": ("2004-05", "2012-13"), "cacp_1221": ("2013-14", "2020-21")}
msp["y0"] = msp["crop_year"].str[:4].astype(int)
keep = msp.apply(lambda r: int(SPAN[r.source_id][0][:4]) <= r.y0 <= int(SPAN[r.source_id][1][:4]), axis=1)
msp = msp[keep].drop_duplicates(["crop", "variety", "crop_year"]).copy()
msp["crop_year"] = msp["crop_year"].replace({"1999-00": "1999-00"})
SEASON = {"paddy": "kharif", "jowar": "kharif", "bajra": "kharif", "ragi": "kharif", "maize": "kharif",
          "arhar": "kharif", "moong": "kharif", "urad": "kharif", "groundnut": "kharif",
          "sunflower": "kharif", "soybean": "kharif", "sesamum": "kharif", "nigerseed": "kharif",
          "cotton": "kharif", "wheat": "rabi", "barley": "rabi", "gram": "rabi", "masur": "rabi",
          "rapeseed_mustard": "rabi", "safflower": "rabi", "toria": "rabi", "jute": "commercial",
          "sugarcane": "commercial"}
GROUP = {"paddy": "rice", "wheat": "wheat", "maize": "maize", "jowar": "millets", "bajra": "millets",
         "ragi": "millets", "arhar": "pulses", "moong": "pulses", "urad": "pulses", "gram": "pulses",
         "masur": "pulses", "groundnut": "oilseeds", "rapeseed_mustard": "oilseeds", "toria": "oilseeds",
         "soybean": "oilseeds", "sunflower": "oilseeds", "sesamum": "oilseeds", "nigerseed": "oilseeds",
         "safflower": "oilseeds", "cotton": "cotton", "sugarcane": "sugarcane", "barley": "other",
         "jute": "other"}
msp["season"] = msp["crop"].map(SEASON)
msp["crop_group"] = msp["crop"].map(GROUP)
msp["unit"] = np.where(msp.crop == "sugarcane", "Rs/quintal (SMP to 2008-09, FRP from 2009-10)", "Rs/quintal")
msp["bonus_flag_in_source"] = msp["raw_token"].str.contains(r"[a-z]$|[^\x00-\x7F]", regex=True)
msp["cost_plus_50_regime"] = (msp["y0"] >= 2018).astype(int)
msp["source_doc"] = msp["source_id"].map(lambda s: SOURCES[s][0])
msp["source_url"] = msp["source_id"].map(lambda s: SOURCES[s][1])
msp = msp.sort_values(["crop", "variety", "y0"]).drop(columns="y0")
msp.to_csv(OUT + "/policy_msp_crop_year.csv", index=False)
cov = msp.pivot_table(index=["crop", "variety"], columns="crop_year", values="msp_rs_per_qtl", aggfunc="first")
print("\nMSP coverage (non-missing years per crop of 24):\n", cov.notna().sum(axis=1).to_string())

gm = pd.DataFrame([(c, GROUP[c], SEASON[c]) for c in GROUP], columns=["msp_crop", "crop_group", "season"])
gm.to_csv(OUT + "/policy_msp_crop_group_map.csv", index=False)

# --- RBI cross-check (RBI series include bonuses; transcribed from the HTML tables)
RBI = {
    ("paddy", "common"): [415, 440, 490, 510, 530, 530, 550, 560, 570, 580, 745, 900, 1050, 1000, 1080, 1250, 1310, 1360, 1410, 1470, 1550, 1750, 1815, 1868],
    ("maize", ""): [360, 390, 415, 445, 485, 485, 505, 515, 525, 540, 620, 840, 840, 880, 980, 1175, 1310, 1310, 1325, 1365, 1425, 1700, 1760, 1850],
    ("wheat", ""): [510, 550, 580, 610, 620, 620, 630, 640, 650, 850, 1000, 1080, 1100, 1170, 1285, 1350, 1400, 1450, 1525, 1625, 1735, 1840, 1925, 1975],
    ("gram", ""): [815, 895, 1015, 1100, 1200, 1220, 1400, 1425, 1435, 1445, 1600, 1730, 1760, 2100, 2800, 3000, 3100, 3175, 3500, 4000, 4400, 4620, 4875, 5100],
    ("arhar", ""): [900, 960, 1105, 1200, 1320, 1320, 1360, 1390, 1400, 1410, 1550, 2000, 2300, 3500, 3700, 3850, 4300, 4350, 4625, 5050, 5450, 5675, 5800, 6000],
    ("moong", ""): [900, 960, 1105, 1200, 1320, 1330, 1370, 1410, 1520, 1520, 1700, 2520, 2760, 3670, 4000, 4400, 4500, 4600, 4850, 5225, 5575, 6975, 7050, 7196],
    ("urad", ""): [900, 960, 1105, 1200, 1320, 1330, 1370, 1410, 1520, 1520, 1700, 2520, 2520, 3400, 3800, 4300, 4300, 4350, 4625, 5000, 5400, 5600, 5700, 6000],
    ("sugarcane", ""): [48, 53, 56, 60, 62, 70, 73, 75, 80, 80, 81, 81, 130, 139, 145, 170, 210, 220, 230, 230, 255, 275, 275, 285],
    ("cotton", "long_staple"): [1530, 1650, 1775, 1825, 1875, 1875, 1925, 1960, 1980, 1990, 2030, 3000, 3000, 3000, 3300, 3900, 4000, 4050, 4100, 4160, 4320, 5450, 5550, 5825],
    ("jute", ""): [570, 650, 750, 785, 810, 850, 860, 890, 910, 1000, 1055, 1250, 1375, 1575, 1675, 2200, 2300, 2400, 2700, 3200, 3500, 3700, 3950, 4225],
    ("groundnut", ""): [980, 1040, 1155, 1220, 1340, 1355, 1400, 1500, 1520, 1520, 1550, 2100, 2100, 2300, 2700, 3700, 4000, 4000, 4030, 4220, 4450, 4890, 5090, 5275],
    ("soybean", "yellow"): [750, 795, 845, 865, 885, 885, 930, 1000, 1010, 1020, 1050, 1390, 1390, 1440, 1690, 2240, 2560, 2560, 2600, 2775, 3050, 3399, 3710, 3880],
    ("sunflower", ""): [1000, 1060, 1155, 1170, 1185, 1195, 1250, 1340, 1500, 1500, 1510, 2215, 2215, 2350, 2800, 3700, 3700, 3750, 3800, 3950, 4100, 5388, 5650, 5885],
    ("rapeseed_mustard", ""): [940, 1000, 1100, 1200, 1300, 1330, 1600, 1700, 1715, 1715, 1800, 1830, 1830, 1850, 2500, 3000, 3050, 3100, 3350, 3700, 4000, 4200, 4425, 4650],
    ("safflower", ""): [910, 990, 1100, 1200, 1300, 1300, 1500, 1550, 1565, 1565, 1650, 1650, 1680, 1800, 2500, 2800, 3000, 3050, 3300, 3700, 4100, 4945, 5215, 5327],
}
ry = [f"{y}-{str(y + 1)[-2:]}" for y in range(1997, 2021)]
xc = []
for (c, v), vals in RBI.items():
    for y, rv in zip(ry, vals):
        mine = msp[(msp.crop == c) & (msp.variety == v) & (msp.crop_year == y)]["msp_rs_per_qtl"]
        xc.append((c, v, y, rv, mine.iloc[0] if len(mine) else np.nan))
xc = pd.DataFrame(xc, columns=["crop", "variety", "crop_year", "rbi_incl_bonus", "compiled"])
xc["diff"] = xc["compiled"] - xc["rbi_incl_bonus"]
xc["rbi_url"] = np.where(xc.crop.isin(["paddy", "maize", "wheat", "gram", "arhar", "moong", "urad"]),
                         SOURCES["rbi_t23"][1], SOURCES["rbi_t24"][1])
xc.to_csv(OUT + "/policy_msp_rbi_crosscheck.csv", index=False)
agree = (xc["diff"].abs() < 1.01).mean()
print("\nRBI cross-check: {} pairs, {:.1%} agree within Rs 1, missing compiled {}".format(
    len(xc), agree, int(xc["compiled"].isna().sum())))
print(xc[(xc["diff"].abs() >= 1.01)].to_string())
