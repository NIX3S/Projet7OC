from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from app.exceptions import validation_exception_handler
from app.api.endpoints import router

app = FastAPI(title="Futurisys ML API")

#ENREGISTRER LE HANDLER AVANT LES ROUTES
app.add_exception_handler(
    RequestValidationError,
    validation_exception_handler
)

#UN SEUL ROUTER
app.include_router(router)