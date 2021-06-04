#!/bin/python3 

import requests

from os import system
from time import sleep

url = "https://status.hpc.rug.nl/"


def still_error(iter = 0):
  try:
    content = requests.get(url).text.split("<li")
    return any("portal.hpc.rug.nl" in service.lower() and "servicestatustag" in service.lower() and not "ok" in service.lower() for service in content)
  except Exception as e:
    if iter < 10:
      still_error(iter + 1)
    else:
      return True


if __name__=="__main__":
  while still_error():
    print("Trying...")
    sleep(900)
  
  if not still_error():
    print("No error.")
    system(f"firefox {url}")