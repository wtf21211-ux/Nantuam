import base64
import os
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException, File, UploadFile, Form, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import auth, credentials
from groq import Groq

# 1. โหลด Environment Variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH", "serviceAccountKey.json")

if not GROQ_API_KEY:
    raise RuntimeError("กรุณาตั้งค่า GROQ_API_KEY ในไฟล์ .env")

# 2. เริ่มต้น Firebase Admin SDK
try:
    cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
    firebase_admin.initialize_app(cred)
except Exception as e:
    print(f"Warning: ไม่สามารถเริ่มต้น Firebase Admin SDK ได้ ({e})")

# 3. เริ่มต้น Groq Client
groq_client = Groq(api_key=GROQ_API_KEY)

# 4. สร้าง FastAPI App
app = FastAPI(
    title="Groq Vision API with Firebase Auth",
    description="Backend service for image analysis using Groq Vision model and Firebase authentication.",
    version="1.0.0"
)

# ตั้งค่า CORS (ปรับแต่ง origin ตามการใช้งานจริง)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Schemes สำหรับรับ Bearer Token จาก Header
security = HTTPBearer()

# ------------------------------------------------------------------
# Auth Dependency: ยืนยันตัวตนด้วย Firebase Token
# ------------------------------------------------------------------
async def verify_firebase_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """
    ตรวจสอบ Firebase ID Token ที่ถูกส่งมาใน Header `Authorization: Bearer <token>`
    คืนค่าเป็น decoded token payload (dict) หากถูกต้อง
    """
    token = credentials.credentials
    try:
        decoded_token = auth.verify_id_token(token)
        return decoded_token
    except auth.ExpiredIdTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Firebase Token หมดอายุ",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except auth.InvalidIdTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Firebase Token ไม่ถูกต้อง",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"การยืนยันตัวตนล้มเหลว: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/")
def read_root():
    return {"message": "Server is running"}


@app.post("/api/v1/analyze-image")
async def analyze_image(
    prompt: str = Form(..., description="คำสั่งหรือคำถามเกี่ยวกับรูปภาพ"),
    image: UploadFile = File(..., description="ไฟล์รูปภาพ (JPEG, PNG, WebP)"),
    model: Optional[str] = Form("llama-3.2-11b-vision-preview", description="ชื่อโมเดล Vision ของ Groq"),
    user_data: dict = Depends(verify_firebase_token)
):
    """
    วิเคราะห์รูปภาพด้วย Groq Vision API (ต้องใส่ Firebase Bearer Token)
    """
    # 1. ตรวจสอบชนิดไฟล์รูปภาพ
    allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
    if image.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"รองรับเฉพาะไฟล์รูปภาพประเภท: {', '.join(allowed_types)}"
        )

    try:
        # 2. อ่านไฟล์รูปภาพและแปลงเป็น Base64 Data URL
        contents = await image.read()
        base64_image = base64.b64encode(contents).decode("utf-8")
        data_url = f"data:{image.content_type};base64,{base64_image}"

        # 3. ส่งคำขอไปยัง Groq API
        chat_completion = groq_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url
                            }
                        }
                    ]
                }
            ],
            temperature=0.2,
            max_completion_tokens=1024,
        )

        response_text = chat_completion.choices[0].message.content

        return {
            "status": "success",
            "user_id": user_data.get("uid"),
            "email": user_data.get("email"),
            "prompt": prompt,
            "model_used": model,
            "result": response_text
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"เกิดข้อผิดพลาดในการประมวลผลรูปภาพ: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
