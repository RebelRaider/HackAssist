import jwt
from fastapi import HTTPException
from .SETTINGS import JWT_KEY, USER_APP_IP, CELLS_APP_IP
import requests
from pydantic import EmailStr


def get_user_info(jwt: str):
    response = requests.get(f"{USER_APP_IP}/me?token={jwt}")
    if response.status_code != 200:
        raise HTTPException(status_code=403, detail="Failed to authenticate user")
    return response.json()


def get_user_id_from_token(jwt_token: str):
    try:
        payload = jwt.decode(jwt_token, JWT_KEY, algorithms=["HS256"])
        user_id = payload.get("user_id")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except:
        raise HTTPException(status_code=401, detail="Invalid token")


def is_worker(jwt: str):
    user_data = get_user_info(jwt)
    role = user_data.get("role")

    if role not in ["Worker", "Admin"]:
        raise HTTPException(
            status_code=403, detail="You don't have permission to process this order"
        )
    return True


def open_cell(cell_id):
    requests.get(f"{CELLS_APP_IP}/{cell_id}/open")


def check_cell_status(cell_id):
    res = requests.get(f"{CELLS_APP_IP}/{cell_id}/status")
    return res.json()


def open_all_cells():
    requests.get(f"{CELLS_APP_IP}/open_all")


def is_admin(jwt: str):
    user_data = get_user_info(jwt)
    role = user_data.get("role")
    if role != "Admin":
        raise HTTPException(status_code=403, detail="You not admin, bro. Go brr")
    return True


def change_role(jwt: str, email: EmailStr, role: str):
    return requests.put(
        f"{USER_APP_IP}/change_role?jwt={jwt}&email={email}&role={role}"
    ).json()
