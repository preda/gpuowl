#!/usr/bin/python3

# Copyright (c) Mihai Preda.
# Inspired by mlucas-primenet.py , part of Mlucas by Ernst W. Mayer.

import argparse
import time
import urllib
import requests
import os
import upload
import getpass

from http import cookiejar
from urllib.parse import urlencode
from urllib.request import build_opener
from urllib.request import HTTPCookieProcessor
from datetime import datetime

baseUrl = "https://www.mersenne.org/"
primenet = build_opener(HTTPCookieProcessor(cookiejar.CookieJar()))

def login(user, password):
    login = {"user_login": user, "user_password": password}
    data = urlencode(login).encode('utf-8')
    r = primenet.open(baseUrl, data).read().decode("utf-8")
    if not user + "<br>logged in" in r:
        print(r)
        print("Login failed");
        raise(PermissionError("Login failed"))

def loadLines(fileName):
    try:
        with open(fileName, 'r') as fi:
            return set((line.strip().strip('\n') for line in fi))
    except FileNotFoundError as e:
        return set()
            
def sendOne(line):
    print("Sending result: ", line)
    data = urlencode({"data": line}).encode('utf-8')
    res = primenet.open(baseUrl + "manual_result/default.php", data).read().decode("utf-8")
    if "Error code" in res:
        begin = res.find("Error code")
        end   = res.find("</div>", begin)
        text = res[begin:end]
        print(text)
        already = text.startswith('Error code: 40, error text: This computer has already sent in this PRP result')
        if already:
            print('Already sent, will not retry')
        return already
    else:
        begin = res.find("CPU credit is")
        end   = res.find("</div>", begin);
        if begin >= 0 and end >= 0:
            print(res[begin:end], '\n')
            return True
        else:
            return False

def appendLine(fileName, line):
    with open(fileName, 'a') as fo: print(line, file = fo, end = '\n')
    
def sendResults(results, sent, sentName, retryName):
    for result in results:
        ok = sendOne(result)
        sent.add(result)
        appendLine(sentName if ok else retryName, result)
        
def fetch(what):
    assignment = {"cores":1, "num_to_get":1, "pref":what}
    # res = primenet.open(baseUrl + "manual_assignment/?" + urlencode(assignment)).read().decode("utf-8")
    res = primenet.open(baseUrl + "manual_assignment/", data=urlencode(assignment).encode()).read().decode("utf-8")
    # print(res)
    
    BEGIN_MARK = "<!--BEGIN_ASSIGNMENTS_BLOCK-->"
    begin = res.find("<!--BEGIN_ASSIGNMENTS_BLOCK-->")
    if begin == -1: raise(AssertionError("assignment no BEGIN mark"))
    begin += len(BEGIN_MARK)
    end   = res.find("<!--END_ASSIGNMENTS_BLOCK-->", begin)
    if end == -1: raise(AssertionError("assignemnt no END mark"))
    line = res[begin:end].strip().strip('\n')
    print(datetime.now(), " New assignment: ", line)
    return line

# LL_DC was 101; here we use 106 instead in order to get only LL_DC with shift != 0 which are the ones
# that can be double-checked with a zero shift.
workTypes = dict(PRP=150, PM1=4, LL_DC=106, PRP_DC=151, PRP_WORLD_RECORD=152, PRP_100M=153, PRP_P1=154)

parser = argparse.ArgumentParser()
parser.add_argument('-u', dest='username', default='', help="Primenet user name")
parser.add_argument('-p', dest='password', help="Primenet password")
parser.add_argument('-t', dest='timeout',  type=int, default=1800, help="Seconds to sleep between updates")
parser.add_argument('--dirs', metavar='DIR', nargs='+', help="GpuOwl directories to scan", default=".")
parser.add_argument('--tasks', dest='nTasks', type=int, default=None, help='Number of tasks to fetch ahead')

choices=list(workTypes.keys())
parser.add_argument('-w', dest='work', choices=choices, help="GIMPS work type", default="PRP")

options = parser.parse_args()
timeout = int(options.timeout)
user = options.username

worktype = workTypes[options.work] if options.work in workTypes else int(options.work)
print("Work type:", worktype)

desiredTasks = options.nTasks if options.nTasks is not None else (12 if worktype == 4 else 2)
print("Will fetch ahead %d tasks. Check every %d sec." % (desiredTasks, timeout))

if not user:
    print("-u USER is required")
    exit(1)
    
print("User: %s" % user)

dirs = [(d if d[-1] == '/' else d + '/' ) for d in options.dirs if d]

print("Watched dirs: ", ' '.join(dirs))

password = options.password
if not password:
    password = getpass.getpass("Primenet password")

# Initial early login, to display any login errors early
login(user, password)

def handle(folder):
    sent = loadLines(folder + "sent.txt")
    (resultsName, worktodoName, sentName, retryName) = (folder + name + ".txt" for name in "results worktodo sent retry".split())
    
    newResults = loadLines(resultsName) - sent
    if newResults: print(datetime.now(), " found %d new result(s) in %s" % (len(newResults), resultsName))
    
    tasks = [line for line in loadLines(worktodoName) if line and line[0] != '#']
    needFetch = len(tasks) < desiredTasks
    if needFetch: print(datetime.now(), " found only %d task(s) in %s, want %d" % (len(tasks), worktodoName, desiredTasks));
    
    if newResults or needFetch:
        login(user, password)
        if newResults: sendResults(newResults, sent, sentName, retryName)
        for _ in range(len(tasks), desiredTasks):
            appendLine(worktodoName, fetch(worktype))

    try:
        os.mkdir(folder + 'uploaded')
    except FileExistsError:
        pass
    
    for entry in os.listdir(folder + 'proof'):
        if entry.endswith('.proof'):
            fileName = folder + 'proof/' + entry
            if upload.uploadProof(user, fileName):
                os.rename(fileName, folder + 'uploaded/' + entry)

while True:
    for folder in dirs:
        try:
            handle(folder)
        except urllib.error.URLError as e:
            print(e)
        except requests.exceptions.ConnectionError as e:
            print(e)
        except requests.exceptions.RequestException as e:
            print(e)

    if timeout == 0:
        break
    
    time.sleep(timeout)
