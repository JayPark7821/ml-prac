import torch
from fastapi import APIRouter

from recomm_app.app.src.model.schemas import RatingRequest
from recomm_app.app.src.model.schemas import RatingResponse

router = APIRouter()


@router.post("/predict/", response_model=RatingResponse)
async def predict_rating(request: RatingRequest) -> RatingResponse:

    return RatingResponse(
        userId=request.userId, movieId=request.movieId, predictedRating=0.0
    )
