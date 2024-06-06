#!/usr/bin/python3

import requests
import hashlib
import sys
import time

def fileBytes(name):
    with open(name, mode='rb') as f:
        return f.read()

def md5(data):
    return hashlib.md5(data).digest().hex()

def headerLines(fileName):
    readLine = lambda f: f.readline().decode().strip()
    with open(fileName, mode='rb') as f:
        return [readLine(f) for _ in range(5)]

def headerExponent(fileName):
    header = headerLines(fileName)
    assert(header[0] == 'PRP PROOF')
    exponent = int(header[4].split('=')[1][1:])
    return exponent

def getNeedRegion(need):
    #print(need)
    firstNeed = sorted([(int(a), b) for a, b in need.items()])[0]
    return firstNeed[0], firstNeed[1] + 1

def uploadChunk(session, baseUrl, pos, chunk):
    url = f'{baseUrl}&DataOffset={pos}&DataSize={len(chunk)}&DataMD5={md5(chunk)}'
    response = session.post(url, files={'Data': (None, chunk)}, allow_redirects=False, timeout=120)
    # headers={'Content-Type:': 'multipart/form-data'}
    # print(f'Response: {response}')
    return response

def upload(userId, exponent, data, verbose):
    fileSize = len(data)
    fileHash = md5(data)
    url = f'https://mersenne.org/proof_upload/?UserID={userId}&Exponent={exponent}&FileSize={fileSize}&FileMD5={fileHash}'
    verbose and print(url)

    while True:
        result = requests.get(url, timeout=20)
        # print(result, result.content, result.headers, result.is_redirect, result.links, result.text, result.url)
        # print("not JSONDecodeError on Result:\n%s\nText:\n%s\n" % (result, result.text))
        
        try:
            json = result.json()
            if 'error_status' in json or 'URLToUse' not in json or 'need' not in json:
              print(json)
              return json['error_status'] == 409 and json['error_description'] == 'Proof already uploaded'
              
            origUrl = json['URLToUse']
            verbose and print(origUrl)
        except json.decoder.JSONDecodeError:
            print("JSONDecodeError on Result:\n%s\nText:\n%s\n" % (result, result.text))
            return False
        
        # baseUrl = 'http' + origUrl[5:] if origUrl.startswith('https:') else origUrl
        baseUrl = 'https' + origUrl[4:] if origUrl.startswith('http:') else origUrl
        if baseUrl != origUrl:
            verbose and print(f'Re-written to: {baseUrl}')

        baseUrl = f'{baseUrl}&FileMD5={fileHash}'
        pos, end = getNeedRegion(json['need'])
        verbose and print(pos, end)

        with requests.session() as session:
            while pos < end:
                size = min(end - pos, 5*1024*1024)
                time1 = time.time()
                response = uploadChunk(session, baseUrl, pos, data[pos:pos+size])
                if response.status_code != 200:
                    print(response)                
                    return False
                pos += size
                time2 = time.time()
                print(f'\r{int(pos/fileSize*100+0.5)}%\t{int(size/(time2 - time1)/1024+0.5)} KB/s    ', end='', flush=True)
                if 'FileUploaded' in response.json():
                    print('')
                    verbose and print('Upload complete')
                    assert(pos >= end)
                    return True

def getTask(userId):
    url = f'http://mersenne.org/oneAssignment/&UserID={userId}&workpref=150'
    print(url)
    r = requests.get(url)
    print(r)
    print(r.json())

def uploadProof(userId, fileName, verbose=False):
    exponent = headerExponent(fileName)
    print(f'Uploading M{exponent} from "{fileName}"')
    data = fileBytes(fileName)
    return upload(userId, exponent, data, verbose)
    
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f'Usage: {sys.argv[0]} <user-id> <proof-file>')
        exit(1)

    userId = sys.argv[1]
    fileName = sys.argv[2]
    if uploadProof(userId, fileName, verbose=True):
        print('Success')
    else:
        exit(1)
