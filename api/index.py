import os
import uuid
import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

app = FastAPI(
    title="QuickMock AI Backend",
    description="AI-powered mock API endpoint generator powered by Gemini",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Supabase Init ────────────────────────────────────────────────────────────
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase_client: Client = None

if supabase_url and supabase_key:
    try:
        supabase_client = create_client(supabase_url, supabase_key)
        print("[OK] Supabase connected.")
    except Exception as e:
        print(f"[ERROR] Supabase init error: {e}")

# ─── Gemini AI Init ───────────────────────────────────────────────────────────
GEMINI_AVAILABLE = False
gemini_client = None

try:
    from google import genai
    gemini_api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if gemini_api_key:
        gemini_client = genai.Client(api_key=gemini_api_key)
        GEMINI_AVAILABLE = True
        print("[OK] Gemini AI initialized.")
    else:
        print("[WARN] GEMINI_API_KEY not set -- AI generation disabled.")
except ImportError:
    print("[WARN] google-genai not installed. Run: pip install google-genai")
except Exception as e:
    print(f"[ERROR] Gemini init error: {e}")


def get_supabase() -> Client:
    """Guard helper — raises 500 if Supabase is not configured."""
    if not supabase_client:
        raise HTTPException(
            status_code=500,
            detail="Database not configured. Set SUPABASE_URL and SUPABASE_KEY in .env."
        )
    return supabase_client


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    """Health check — reports DB and AI status to the frontend."""
    return {
        "status": "healthy",
        "database": "connected" if supabase_client else "missing_credentials",
        "ai": "available" if GEMINI_AVAILABLE else "missing_api_key",
    }


@app.post("/api/generate-ai-json")
async def generate_ai_json(request: Request):
    """
    Use Gemini AI to generate realistic mock JSON from a natural language prompt.
    Returns { success: true, json: "<formatted json string>" }
    """
    if not GEMINI_AVAILABLE or not gemini_client:
        raise HTTPException(
            status_code=503,
            detail="Gemini AI is not configured. Add GEMINI_API_KEY to your .env file and restart the server."
        )

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON.")

    prompt = body.get("prompt", "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    if len(prompt) > 600:
        raise HTTPException(status_code=400, detail="Prompt too long. Maximum 600 characters.")

    try:
        from google.genai import types
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"Generate realistic mock API response JSON for: {prompt}",
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are a professional API mock data generator for developers. "
                    "Your ONLY output must be raw, valid JSON — either an object { } or an array [ ]. "
                    "NEVER include markdown code fences (```), backticks, explanations, comments, "
                    "or any text outside of the JSON structure. "
                    "The very first character of your response must be { or [. "
                    "Generate realistic, diverse, and developer-friendly data."
                )
            )
        )
        raw = response.text.strip()

        # Defensive: strip any accidental markdown code fences
        if "```" in raw:
            parts = raw.split("```")
            for part in parts:
                cleaned = part.strip()
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
                if cleaned.startswith("{") or cleaned.startswith("["):
                    raw = cleaned
                    break

        raw = raw.strip()

        # Validate the output is actually JSON
        parsed = json.loads(raw)
        formatted = json.dumps(parsed, indent=2)

        return {"success": True, "json": formatted}

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=422,
            detail="AI returned invalid JSON. Please rephrase your prompt with more specific details."
        )
    except Exception as e:
        msg = str(e)
        if "SAFETY" in msg.upper() or "blocked" in msg.lower():
            raise HTTPException(
                status_code=400,
                detail="Prompt was flagged by safety filters. Please use a different description."
            )
        raise HTTPException(
            status_code=500,
            detail=f"AI generation failed. Please try again. ({msg[:200]})"
        )


@app.post("/api/create")
async def create_mock_endpoint(request: Request):
    """
    Save a JSON payload to Supabase and return its UUID as a live endpoint ID.
    """
    db = get_supabase()

    try:
        json_payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON.")

    try:
        result = db.table("mock_endpoints").insert({"json_data": json_payload}).execute()
        if not result.data:
            raise HTTPException(status_code=500, detail="Database insert returned no data.")
        record = result.data[0]
        return {
            "success": True,
            "id": record["id"],
            "created_at": record.get("created_at"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/api/mock/{mock_id}")
async def get_mock_endpoint(mock_id: str):
    """
    Retrieve a stored JSON mock payload by its UUID.
    """
    try:
        uuid.UUID(mock_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID format.")

    db = get_supabase()

    try:
        result = db.table("mock_endpoints").select("json_data").eq("id", mock_id).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="Mock endpoint not found.")
        return JSONResponse(content=result.data[0]["json_data"])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
