from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

app = FastAPI()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []

    for error in exc.errors():
        field_name = error["loc"][-1]
        if error["type"] == "missing":
            # Champ manquant
            errors.append(f"Feature manquante : {field_name}")
        elif error["type"].startswith("type_error") or error["type"].startswith("value_error"):
            # Type incorrect
            errors.append(f"Type incorrect pour {field_name} : {error['msg']}")
        else:
            # Autres erreurs
            errors.append(f"{field_name} : {error['msg']}")

    return JSONResponse(
        status_code=422,
        content={"detail": errors},
    )