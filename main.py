import os
import json
import base64
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
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
        return {"reply": "❌ เกิดข้อผิดพลาด: ยังไม่ได้ตั้งค่า GROQ_API_KEY ใน Vercel Environment Variables"}

    try:
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
        
        # 1. กรณีมีไฟล์ภาพแนบมา -> ใช้ Vision Model
        if file and file.filename:
            contents = await file.read()
            base64_image = base64.b64encode(contents).decode('utf-8')
            mime_type = file.content_type or "image/jpeg"
            image_url = f"data:{mime_type};base64,{base64_image}"
            
            prompt_text = message if message.strip() else "ช่วยวิเคราะห์ภาพนี้ว่ามีปัญหาน้ำท่วมหรือความเสี่ยงอะไรไหม ถ้ามีต้องแก้ไขอย่างไร ถ้าไม่มีให้แจ้งว่าปกติครับ"

            messages_payload = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        }
                    ]
                }
            ]

            completion = client.chat.completions.create(
                model="llama-3.2-11b-vision-preview",
                messages=messages_payload,
                temperature=0.5,
                max_tokens=1024
            )

        # 2. กรณีพิมพ์ข้อความอย่างเดียว (เช่น พิมพ์ "Hi") -> ใช้ Llama 3.3 70B
        elif message and message.strip():
            messages_payload = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message.strip()}
            ]

            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages_payload,
                temperature=0.7,
                max_tokens=1024
            )
            
        else:
            return {"reply": "กรุณาพิมพ์ข้อความหรือแนบรูปภาพส่งเข้ามาครับ"}

        # ตรวจสอบการส่งกลับของข้อมูล
        if completion.choices and len(completion.choices) > 0:
            response_text = completion.choices[0].message.content
            return {"reply": response_text}
        else:
            return {"reply": "AI ไม่ได้ตอบกลับข้อมูล กรุณาลองใหม่อีกครั้ง"}

    except Exception as e:
        # หากเกิด Error ฝั่ง AI ให้ส่งรายละเอียด Error กลับไปโชว์ในแชทแทน undefined
        error_msg = str(e)
        print(f"Groq API Error: {error_msg}")
        return {"reply": f"⚠️ เกิดข้อผิดพลาดจาก AI: {error_msg}"}
        
