import requests

response = requests.post("https://fastapisamba-dc4b94e92e38.herokuapp.com/predict", json={
  "model_key": "Audi",
  "mileage":151555,
  "engine_power":230,
  "fuel": "diesel",
  "paint_color": "blue",
  "car_type": "estate",
  "private_parking_available": True,
  "has_gps": True,
  "has_air_conditioning": True,
  "automatic_car": False,
  "has_getaround_connect": True,
  "has_speed_regulator": False,
  "winter_tires": True
  })
print(response.json())