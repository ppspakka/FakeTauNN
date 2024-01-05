import os
import json

from taus import extract_taus

# for Local run
#root = "/mnt/c/Users/Thammachath/Desktop/Code/Project/raw_data/NanoAOD_v9/"

# for CU e-Science run
root = "/work/project/cms/thammachath/NanoAOD_v9/"

ttbar_training_files = [
    "143F7726-375A-3D48-9D53-D6B071CED8F6.root",
    "15FC5EA3-70AA-B640-8748-BD5E1BB84CAC.root",
    "1CD61F25-9DE8-D741-9200-CCBBA61E5A0A.root",
    "1D885366-E280-1243-AE4F-532D326C2386.root",
    "23AD2392-C48B-D643-9E16-C93730AA4A02.root",
    "245961C8-DE06-8F4F-9E92-ED6F30A097C4.root",
    "262EAEE2-14CC-2A44-8F4B-B1A339882B25.root",
    "2EEEF2A2-D775-764F-8ED6-EF0D5B425739.root",
    "329FB0B6-F45B-8D4B-A27C-3D61E33C91DC.root",
    "3757682B-9F48-3B44-88CC-632400689053.root",
    "3EA6E929-788C-B94F-84B6-1855A7DBB589.root",
    "42735D52-F6A9-5D4B-A919-9620DA7331DB.root",
    "43FD54B7-8F80-8648-B72C-44A065727306.root",
    "44E3579B-F232-834C-AAE1-79B46AB34D41.root",
    "52557D67-72C0-754B-AC60-7AEE7FF1FD4A.root",
    "54CC88B4-7C8E-4249-ADE7-3969965304E2.root",
    "5FA524E5-403F-5746-8382-08EE22417B0D.root"
]


file_paths = [os.path.join(root, f) for f in ttbar_training_files]

if not os.path.exists(os.path.join(os.path.dirname(__file__), "dataset")):
    os.mkdir(os.path.join(os.path.dirname(__file__), "dataset"))

extracted = [os.path.join("dataset", f"MTaus_{i}.root") for i in range(len(file_paths))]

d = {
    "RECOTAU_RECOJET": (0, 0),
}

if __name__ == "__main__":
    for file_in, file_out in zip(file_paths, extracted):
        extract_taus(file_in, file_out, d)

    with open(os.path.join(os.path.dirname(__file__), "match_dict.json"), "w") as f:
        json.dump(d, f)
