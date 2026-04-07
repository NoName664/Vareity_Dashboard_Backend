from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware

import json
import gzip
import numpy as np
import polars as pl
import os
import datetime
from firebase_admin import credentials, initialize_app, storage, auth, firestore
import csv
import io

# ==============================
# FASTAPI APP
# ==============================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# FIREBASE LAZY INIT
# ==============================

firebase_app = None
db = None


def get_db():
    global firebase_app, db

    if db is None:

        cred = credentials.Certificate("./account_key.json")

        firebase_app = initialize_app(
            cred,
            {"storageBucket": "rcef-data.firebasestorage.app"}
        )

        db = firestore.client()

    return db


def get_bucket():
    get_db()
    return storage.bucket("rcef-data.firebasestorage.app")

# ==============================
# CACHE SYSTEM
# ==============================

CACHE = {}
CACHE_DATE = datetime.date.today()


def check_cache_reset():
    global CACHE, CACHE_DATE

    today = datetime.date.today()

    if today != CACHE_DATE:
        CACHE.clear()
        CACHE_DATE = today

# ==============================
# UTILITIES
# ==============================


def get_signed_url(bucket_name, file_path):

    bucket = storage.bucket(bucket_name)

    blob = bucket.blob(file_path)

    return blob.generate_signed_url(
        expiration=datetime.timedelta(hours=1),
        method="GET"
    )


def compress_json_response(data):

    json_data = json.dumps(data)

    return gzip.compress(json_data.encode("utf-8"))

# ==============================
# FIREBASE FILE READERS
# ==============================


def read_csv_from_firebase():

    check_cache_reset()

    if "VarietyMapping" in CACHE:
        return CACHE["VarietyMapping"]

    bucket = get_bucket()

    blob = bucket.blob("VarietyMapping.csv")

    data = blob.download_as_bytes()

    df = pl.read_csv(io.BytesIO(data))

    CACHE["VarietyMapping"] = df

    return df


def read_library_parquet_from_firebase():

    check_cache_reset()

    if "Library" in CACHE:
        return CACHE["Library"]

    bucket = get_bucket()

    blob = bucket.blob("Library.parquet")

    data = blob.download_as_bytes()

    df = pl.read_parquet(io.BytesIO(data))

    CACHE["Library"] = df

    return df


def read_parquet_from_firebase(file_name):

    check_cache_reset()

    if file_name in CACHE:
        return CACHE[file_name]

    bucket = get_bucket()

    blob = bucket.blob(f"data/{file_name}.parquet")

    data = blob.download_as_bytes()

    df = pl.read_parquet(io.BytesIO(data))

    CACHE[file_name] = df

    return df


def read_parquet_from_firebase_harvest():

    check_cache_reset()

    if "HarvestWeek" in CACHE:
        return CACHE["HarvestWeek"]

    bucket = get_bucket()

    blob = bucket.blob("HarvestWeek.parquet")

    data = blob.download_as_bytes()

    df = pl.read_parquet(io.BytesIO(data))

    CACHE["HarvestWeek"] = df

    return df


def read_parquet_from_firebase_harvest_1():

    check_cache_reset()

    if "HarvestWeek_1" in CACHE:
        return CACHE["HarvestWeek_1"]

    bucket = get_bucket()

    blob = bucket.blob("HarvestWeek_1.parquet")

    data = blob.download_as_bytes()

    df = pl.read_parquet(io.BytesIO(data))

    CACHE["HarvestWeek_1"] = df

    return df


def read_parquet_from_firebase_maturity():

    check_cache_reset()

    if "MaturityWeek" in CACHE:
        return CACHE["MaturityWeek"]

    bucket = get_bucket()

    blob = bucket.blob("MaturityWeek.parquet")

    data = blob.download_as_bytes()

    df = pl.read_parquet(io.BytesIO(data))

    CACHE["MaturityWeek"] = df

    return df

def read_parquet_from_firebase_ds():

    check_cache_reset()

    if "DS_Week" in CACHE:
        return CACHE["DS_Week"]

    bucket = get_bucket()

    blob = bucket.blob("DS_Week.parquet")

    data = blob.download_as_bytes()

    df = pl.read_parquet(io.BytesIO(data))

    CACHE["DS_Week"] = df

    return df


def read_parquet_from_firebase_ws():

    check_cache_reset()

    if "WS_Week" in CACHE:
        return CACHE["WS_Week"]

    bucket = get_bucket()

    blob = bucket.blob("WS_Week.parquet")

    data = blob.download_as_bytes()

    df = pl.read_parquet(io.BytesIO(data))

    CACHE["WS_Week"] = df

    return df

# ==============================
# REGION / PROVINCE MAPPING
# ==============================


def load_region_selection_mapping() -> dict[str, dict[str, list[str]]]:

    check_cache_reset()

    if "region_mapping" in CACHE:
        return CACHE["region_mapping"]

    mapping: dict[str, dict[str, list[str]]] = {}

    bucket = get_bucket()

    blob = bucket.blob("Geographic.csv")

    csv_data = blob.download_as_text(encoding="utf-8-sig")

    reader = csv.DictReader(io.StringIO(csv_data))

    for row in reader:

        region = row["Region"].strip()
        province = row["Province"].strip()
        cmlgu = row["CMLGU"].strip()

        mapping.setdefault(region, {})
        mapping[region].setdefault(province, [])

        if cmlgu not in mapping[region][province]:
            mapping[region][province].append(cmlgu)

    CACHE["region_mapping"] = mapping

    return mapping


def load_province_mapping():

    check_cache_reset()

    if "province_mapping" in CACHE:
        return CACHE["province_mapping"]

    mapping = {}

    bucket = get_bucket()

    blob = bucket.blob("Muni/province_mapping.csv")

    csv_data = blob.download_as_text(encoding="utf-8-sig")

    reader = csv.DictReader(io.StringIO(csv_data))

    for row in reader:
        mapping[row["Province"].strip()] = row["PSGC_Code"].strip()

    CACHE["province_mapping"] = mapping
    
    return mapping

# ==============================
# API 1
# ==============================


@app.get("/list-data-files")
async def list_data_files():

    bucket = get_bucket()

    blobs = list(bucket.list_blobs(prefix="data/"))

    files = []

    for blob in blobs:

        if blob.name.endswith(".parquet"):

            name = blob.name.replace("data/", "").replace(".parquet", "")

            if name.startswith("ds"):
                label = f"{name[2:6]} Dry Season"
            elif name.startswith("ws"):
                label = f"{name[2:6]} Wet Season"
            else:
                label = name

            files.append({"file": name, "label": label})

    return {"files": files}

# ==============================
# API 2
# ==============================


@app.post("/process_file")
async def process_file(request: Request):

    try:

        data = await request.json()

        file_name = data.get("file")

        if not file_name:
            return JSONResponse({"error": "file parameter required"}, 400)

        df = read_parquet_from_firebase(file_name)

        df = df.rename({
            "Old_MuniCode": "OldPSGC",
            "SeedVariety": "variety_0",
            "PlantingWeek": "planting_week"
        })

        df = df.with_columns(
            pl.when(pl.col("Sex") == "MALE").then(pl.lit("MEN"))
            .when(pl.col("Sex") == "FEMALE").then(pl.lit("WOMEN"))
            .otherwise(pl.col("Sex"))
            .alias("Sex")
        )

        df_psgc = read_library_parquet_from_firebase().rename({
            "Old PSGC": "OldPSGC"
        })

        df_variety = read_csv_from_firebase()

        df = df.join(df_psgc, on="OldPSGC", how="left")
        df = df.join(df_variety, on="variety_0", how="left")

        df_week = None
        if "ds" in file_name:
            df_week = read_parquet_from_firebase_ds()
        elif "ws" in file_name:
            df_week = read_parquet_from_firebase_ws()

        if df_week is not None:

            df = df.with_columns(pl.col("planting_week").cast(pl.Utf8))
            df_week = df_week.with_columns(pl.col("planting_week").cast(pl.Utf8))

            df_week = df_week.rename({
                "Planting Week": "Planting_Week"
            })

            df = df.join(df_week, on="planting_week", how="left")

        else:
            df = df.with_columns(pl.lit("Unknown").alias("Planting_Week"))

        df_harvest = read_parquet_from_firebase_harvest().rename({
            "Weeks": "Planting_Week"
        })

        df = df.join(df_harvest, on="Planting_Week", how="left")

        df_maturity = read_parquet_from_firebase_maturity().rename({
            "Variety": "corrected_variety"
        })

        df = df.join(df_maturity, on="corrected_variety", how="left")

        df = df.with_columns(
            (((pl.col("Sort") + pl.col("MaturityWeek") - 1) % 48) + 1)
            .alias("HarvestWeek")
        )

        df_harvest_weeks = read_parquet_from_firebase_harvest_1().rename({
            "HarvestWeeks": "HarvestWeek"
        })

        df = df.join(df_harvest_weeks, on="HarvestWeek", how="left")

        required_columns = [
            'Region', 'Province', 'CMLGU', 'Barangay',
            'OldPSGC', 'PSGC', 'Sex',
            'Ecosystem', 'CropEstablishment',
            'Planting_Week', 'Weeks',
            'corrected_variety',
            'TotalArea', 'Farmers_count', 'ClaimedBags'
        ]

        df = df.select(required_columns)

        response_data = df.to_dicts()

        compressed = compress_json_response(response_data)

        return Response(
            content=compressed,
            media_type="application/json",
            headers={"Content-Encoding": "gzip"}
        )

    except Exception as e:

        return JSONResponse({"error": str(e)}, 500)

# ==============================
# API 3
# ==============================


@app.api_route("/geojson-region", methods=["GET", "POST"])
async def get_geojson_region(request: Request):

    if request.method == "POST":
        data = await request.json()
        region_name = data.get("region_name")
    else:
        region_name = request.query_params.get("region_name")

    bucket = get_bucket()

    region_to_psgc = {
        'Ilocos': 'PH010000000',
        'Cagayan Valley': 'PH020000000',
        'Central Luzon': 'PH030000000',
        'CALABARZON': 'PH040000000',
        'MIMAROPA': 'PH170000000',
        'Bicol': 'PH050000000',
        'Western Visayas': 'PH060000000',
        'Central Visayas': 'PH070000000',
        'Eastern Visayas': 'PH080000000',
        'Zamboanga Peninsula': 'PH090000000',
        'Northern Mindanao': 'PH100000000',
        'Davao': 'PH110000000',
        'SOCCSKSARGEN': 'PH120000000',
        'Caraga': 'PH160000000',
        'BARMM': 'PH150000000',
        'CAR': 'PH140000000',
        'NCR': 'PH130000000',
        'NIR': 'PH180000000'
    }

    if region_name == "All":

        features = []

        for name, psgc in region_to_psgc.items():

            blob = bucket.blob(f"Province/{psgc}.json")

            if not blob.exists():
                continue

            geojson = json.loads(blob.download_as_text())

            for f in geojson["features"]:
                f["properties"]["region_name"] = name

            features.extend(geojson["features"])

        return {"type": "FeatureCollection", "features": features}

    url = get_signed_url(
        "rcef-data.firebasestorage.app",
        f"Province/{region_to_psgc[region_name]}.json"
    )

    return {"geojson_url": url}

# ==============================
# API 4
# ==============================


@app.post("/geojson-province")
async def get_geojson_province(request: Request):

    try:

        data = await request.json()

        if not data or "province_name" not in data:
            return JSONResponse(
                {"error": "Province name is required"},
                status_code=400
            )

        province = data["province_name"].strip()

        mapping = load_province_mapping()

        psgc = mapping.get(province)

        if not psgc:
            return JSONResponse(
                {"error": "Invalid province name"},
                status_code=400
            )

        bucket_name = "rcef-data.firebasestorage.app"
        bucket = storage.bucket(bucket_name)

        file_path = f"Muni/{psgc}.json"

        blob = bucket.blob(file_path)

        if not blob.exists():
            return JSONResponse(
                {"error": "GeoJSON file not found"},
                status_code=404
            )

        geojson_url = get_signed_url(bucket_name, file_path)

        return {"geojson_url": geojson_url}

    except Exception as e:

        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )

# ==============================
# API 5
# ==============================


@app.post("/geojson-municipality")
async def get_geojson_municipality(request: Request):

    data = await request.json()

    psgc = data.get("psgc_code")

    url = get_signed_url(
        "rcef-data.firebasestorage.app",
        f"Brgy/{psgc}.json"
    )

    return {"geojson_url": url}

# ==============================
# API 6
# ==============================


@app.get("/regions")
async def get_regions():

    mapping = load_region_selection_mapping()

    return sorted(mapping.keys())

# ==============================
# API 7
# ==============================


@app.get("/provinces")
async def get_provinces(region: str) -> list[str]:

    mapping = load_region_selection_mapping()
    provinces_dict = mapping.get(region)
    
    if not provinces_dict:
        return []

    return sorted(provinces_dict.keys())

# ==============================
# API 8
# ==============================


@app.get("/municipalities")
async def get_municipalities(region: str, province: str) -> list[str]:

    mapping = load_region_selection_mapping()
    provinces_dict = mapping.get(region)
    
    if not provinces_dict:
        return []
        
    municipalities_list = provinces_dict.get(province)
    if not municipalities_list:
        return []

    return sorted(municipalities_list)

# ==============================
# API 9
# ==============================


@app.post("/register-user")
async def register_user(request: Request):

    data = await request.json()

    user = auth.create_user(
        email=data["email"].strip(),
        password=data["password"]
    )

    uid = user.uid

    get_db().collection("users").document(uid).set({
        "uid": uid,
        "firstName": data.get("firstName"),
        "lastName": data.get("lastName"),
        "occupation": data.get("occupation"),
        "gender": data.get("gender"),
        "birthdate": data.get("birthdate"),
        "region": data.get("region"),
        "province": data.get("province"),
        "municipality": data.get("municipality"),
        "role": data.get("role", "User"),
        "createdAt": firestore.SERVER_TIMESTAMP
    })

    return {"message": "User registered successfully", "uid": uid}

# ==============================
# API 10
# ==============================


@app.get("/me")
async def get_logged_user(request: Request):

    auth_header = request.headers.get("Authorization")
    print("[DEBUG] /me Authorization header:", auth_header)

    if not auth_header or not auth_header.startswith("Bearer "):
        print("[DEBUG] /me unauthorized request")
        return JSONResponse({"error": "Unauthorized: missing or invalid Authorization header"}, 401)

    token = auth_header.split("Bearer ", 1)[1].strip()

    try:
        decoded = auth.verify_id_token(token)
    except Exception as e:
        return JSONResponse({"error": f"Unauthorized: invalid token ({str(e)})"}, 401)

    uid = decoded.get("uid")
    if not uid:
        return JSONResponse({"error": "Unauthorized: token missing uid"}, 401)

    user_doc = get_db().collection("users").document(uid).get()
    if not user_doc.exists:
        return JSONResponse({"error": "User not found"}, 404)

    user = user_doc.to_dict() or {}
    firstName = f"{user.get('firstName','')}".strip()
    lastName = f"{user.get('lastName','')}".strip()
    gender = f"{user.get('gender')}".strip()
    birthdate = f"{user.get('birthdate')}".strip()
    occupation = f"{user.get('occupation')}".strip()
    municipality = f"{user.get('municipality')}".strip()
    province = f"{user.get('province','')}".strip()
    region = f"{user.get('region','')}".strip()
    role = f"{user.get('role', 'User')}".strip()
    createdAt = f"{user.get('createdAt','')}".strip()

    return {"firstName": firstName,
            "lastName": lastName,
            "gender": gender,
            "birthdate": birthdate,
            "occupation": occupation,
            "municipality": municipality,
            "province": province,
            "region": region,
            "role": role,
            "createdAt": createdAt}

# ==============================
# API 11
# ==============================


@app.post("/update-profile")
async def update_profile(request: Request):

    auth_header = request.headers.get("Authorization")

    if not auth_header or not auth_header.startswith("Bearer "):
        return JSONResponse({"error": "Unauthorized"}, 401)

    token = auth_header.split("Bearer ", 1)[1].strip()

    try:
        decoded = auth.verify_id_token(token)
    except Exception as e:
        return JSONResponse({"error": f"Unauthorized: invalid token ({str(e)})"}, 401)

    uid = decoded.get("uid")
    if not uid:
        return JSONResponse({"error": "Unauthorized: token missing uid"}, 401)

    data = await request.json()

    allowed_fields = [
        "firstName", "lastName", "gender", "birthdate",
        "occupation", "region", "province", "municipality"
    ]

    update_data = {k: v for k, v in data.items() if k in allowed_fields}

    if not update_data:
        return JSONResponse({"error": "No valid fields to update"}, 400)

    get_db().collection("users").document(uid).update(update_data)

    return {"message": "Profile updated successfully", "updated": update_data}

# ==============================
# API 12
# ==============================

@app.post("/update-user")
async def update_user(request: Request):

    data = await request.json()

    uid = data.get("uid")
    if not uid:
        return JSONResponse({"error": "uid is required"}, 400)

    allowed_fields = [
        "firstName", "lastName", "gender", "birthdate",
        "occupation", "region", "province", "municipality"
    ]

    update_data = {k: v for k, v in data.items() if k in allowed_fields}

    if not update_data:
        return JSONResponse({"error": "No valid fields to update"}, 400)

    user_doc = get_db().collection("users").document(uid).get()
    if not user_doc.exists:
        return JSONResponse({"error": "User not found"}, 404)

    get_db().collection("users").document(uid).update(update_data)

    return {"message": "User updated successfully", "updated": update_data}

# ==============================
# API 13
# ==============================

@app.get("/users")
async def get_all_users():
    try:
        users_ref = get_db().collection("users")
        docs = users_ref.stream()

        users_list = []
        for doc in docs:
            user_data = doc.to_dict()
            users_list.append(user_data)
        
        return {"users": users_list}
    
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)

# ==============================
# CLOUD RUN SERVER
# ==============================

if __name__ == "__main__":

    import uvicorn

    port = int(os.environ.get("PORT", 8080))

    uvicorn.run(app, host="0.0.0.0", port=port)