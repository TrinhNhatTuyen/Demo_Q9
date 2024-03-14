import requests, datetime, hashlib
# def get_accesstoken(client_id='87ed6cf1e9274e65af6500193fd7dce8', 
#                     clientsecret='5e56225a865fc7368f7e1e57b5bdd0fc', 
#                     username='trinhnhattuyen12a4@gmail.com', 
#                     password='nhattuyen0414'):
def get_accesstoken(client_id='2ce5129232f74cc2ac89e24cdd04ec65', 
                    clientsecret='5fc0bb10aa78a6fa48c0ff4b95e3c791', 
                    username='datlongan@gmail.com', 
                    password='Dat12345678'):
    print("Get access token ...")
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    url = "https://euapi.sciener.com/oauth2/token"
    
    data = {
        "clientId": client_id,
        "clientSecret": clientsecret,
        "username": username,
        "password": hashlib.md5(password.encode()).hexdigest()
    }
    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        print("Get access token successful !")
        return response.json()['access_token']
    else:
        return "Failed to get access token"

def lock(access_token, lock_id, client_id='2ce5129232f74cc2ac89e24cdd04ec65'):
    # access_token = get_accesstoken()
    seconds = 2
    url = "https://euapi.sciener.com/v3/lock/lock"
    
    now = datetime.datetime.now()
    new_time = now + datetime.timedelta(seconds=seconds)
    timestamp = int(new_time.timestamp() * 1000)
    
    data = {
        "clientId": client_id,
        "accessToken": access_token,
        "lockId": lock_id,
        "date": timestamp
    }
    response = requests.post(url, data=data)
    if response.status_code == 200:
        print("Lock successful !")
    else:
        print("Lock failed")
    
def unlock(access_token, lock_id, client_id='2ce5129232f74cc2ac89e24cdd04ec65'):
    # access_token = get_accesstoken()
    seconds = 2
    url = "https://euapi.sciener.com/v3/lock/unlock"
    
    now = datetime.datetime.now()
    new_time = now + datetime.timedelta(seconds=seconds)
    timestamp = int(new_time.timestamp() * 1000)
    
    data = {
        "clientId": client_id,
        "accessToken": access_token,
        "lockId": lock_id,
        "date": timestamp
    }
    response = requests.post(url, data=data)
    if response.status_code == 200 and response.json()['errcode']==0:
        print("Unlock successful !")
    else:
        print("Unlock failed")
        
# print(get_accesstoken())
