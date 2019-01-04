#!/usr/bin/python3

import sys
import os.path
import re
from time import sleep
import os
from optparse import OptionParser

#import http.cookiejar as cookiejar
from http import cookiejar
#from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import build_opener
from urllib.request import HTTPCookieProcessor

baseUrl = "https://www.mersenne.org/"
primenet = build_opener(HTTPCookieProcessor(cookiejar.CookieJar()))

def login(user, password):
    login = {"user_login": user, "user_password": password}
    data = urlencode(login).encode('utf-8')
    r = primenet.open(baseUrl + "default.php", data).read().decode("utf-8")
    if not user + "<br>logged in" in r:
        print("Login failed");
        raise(PermissionError("Login failed"))

def loadLines(fileName):
    try:
        with open(fileName, 'r') as fi:
            return set((line.strip().strip('\n') for line in fi))
    except FileNotFoundError as e:
        return set()
            
def sendOne(line):
    print(line)
    data = urlencode({"data": line}).encode('utf-8')
    res = primenet.open(baseUrl + "manual_result/default.php", data).read().decode("utf-8")
    if "Error code" in res:
        begin = res.find("Error code")
        end   = res.find("</div>", begin);
        print(res[begin:end])
    else:
        print(res);
    print()
        
def sendResults(results):
    with open('sent.txt', 'a') as sent:
        for result in results:
            sendOne(result)
            print(result, file=sent, end='\n')
            sentResults.add(result)
    
def fetch(what):
    assignment = {"cores":1, "num_to_get":1, "pref":what}
    #print(urlencode(assignment))    
    res = primenet.open(baseUrl + "manual_assignment/?" + urlencode(assignment)).read().decode("utf-8")
    #print(res)
    BEGIN_MARK = "<!--BEGIN_ASSIGNMENTS_BLOCK-->"
    begin = res.find("<!--BEGIN_ASSIGNMENTS_BLOCK-->")
    if begin == -1: raise(AssertionError("assignment no BEGIN mark"))
    begin += len(BEGIN_MARK)
    end   = res.find("<!--END_ASSIGNMENTS_BLOCK-->", begin)
    if end == -1: raise(AssertionError("assignemnt no END mark"))
    line = res[begin:end].strip().strip('\n')
    print(line)
    return line

PRP_FIRST_TIME = 150
PRP_DC = 151

user = "preda"
password = input()
sentResults = loadLines("sent.txt")

while True:
    newResults = loadLines("results.txt") - sentResults
    print("found %d new results" % len(newResults))
    
    tasks = [line for line in loadLines("worktodo.txt") if line and line[0] != '#']
    print("found %d worktodo.txt tasks" % len(tasks));
    
    if newResults or len(tasks) < 2: login(user, password)
    if newResults: sendResults(newResults)
    
    if len(tasks) < 2:
        with open("worktodo.txt", "a") as fo: print(fetch(PRP_FIRST_TIME), file=fo, end='\n')
    sleep(5)
