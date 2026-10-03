import os
import json
import base64
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from dotenv import load_dotenv

# โหลด Environment Variables
load_dotenv()

app = FastAPI()

# อนุญาตให้ Frontend เชื่อมต่อเข้ามาได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ดึง GROQ_API_KEY จาก Environment Variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

@app.get("/")
def read_root():
    return {"status": "Flood AI Backend is running"}

@app.post("/chat-analyze")
async def chat_analyze(
    message: str = Form(""),
    file: UploadFile = File(None)
):
    if not client:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured in Environment Variables")

    try:
        messages_payload = []
        
        system_prompt = """
        คุณคือ 'Flood Safety AI' ผู้เชี่ยวชาญด้านประเมินภัยพิบัติน้ำท่วมและความปลอดภัย
        
        หลักการตอบ:
        1. ถ้ามีภาพ:
           - ตรวจสอบความเสี่ยงน้ำท่วม/จุดอันตราย
           - ถ้า 'มีปัญหา': ระบุระดับความเสี่ยง (CRITICAL/HIGH/MEDIUM/LOW), ระดับน้ำโดยประมาณ และวิธีแก้/สิ่งที่ต้องทำทันที
           - ถ้า 'ไม่มีปัญหา': บอกว่าสถานการณ์ปกติ ไม่พบความเสี่ยง และแนะนำการเฝ้าระวังทั่วไป
        2. ถ้าเป็นข้อความพิมพ์คุย:
           - ให้ตอบคำถามเกี่ยวกับน้ำท่วม การเตรียมตัว และความปลอดภัยอย่างกระชับ ชัดเจน
        """
        
        messages_payload.append({"role": "system", "content": system_prompt})
        user_content = []
        
        if message:
            user_content.append({"type": "text", "text": message})

        if file:
            contents = await file.read()
            base64_image = base64.b64encode(contents).decode('utf-8')
            image_url = f"data:{file.content_type};base64,{base64_image}"
            
            user_content.append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })
            
            if not message:
                user_content.append({
                    "type": "text", 
                    "text": "ช่วยวิเคราะห์ภาพนี้ว่ามีปัญหาน้ำท่วมหรือความเสี่ยงอะไรไหม ถ้ามีต้องแก้ไขอย่างไร ถ้าไม่มีให้แจ้งว่าปกติครับ"
                })

        if not user_content:
            raise HTTPException(status_code=400, detail="กรุณาส่งภาพหรือข้อความอย่างใดอย่างหนึ่ง")

        messages_payload.append({"role": "user", "content": user_content})

        completion = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=messages_payload,
            temperature=0.5,
            max_tokens=1024
        )

        response_text = completion.choices[0].message.content
        return {"reply": response_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
