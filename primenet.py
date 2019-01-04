#!/usr/bin/python3

# Copyright (c) Mihai Preda.
# Inspired by mlucas-primenet.py , part of Mlucas by Ernst W. Mayer.

from time import sleep
from optparse import OptionParser
from http import cookiejar
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
    print("Sending result: ", line)
    data = urlencode({"data": line}).encode('utf-8')
    res = primenet.open(baseUrl + "manual_result/default.php", data).read().decode("utf-8")
    if "Error code" in res:
        begin = res.find("Error code")
        end   = res.find("</div>", begin)
        print(res[begin:end], '\n')
        return False
    else:
        print(res, '\n')
        return True
        
def sendResults(results):
    with open('sent.txt', 'a') as sent, open('retry.txt', 'a') as retry:
        for result in results:
            if not sendOne(result):
                print(result, file=retry, end='\n')
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

parser = OptionParser()
parser.add_option('-u', '--username', dest='username', default='', help="Primenet user name")
parser.add_option('-p', '--password', dest='password', help="Primenet password")
parser.add_option('-t', '--timeout',  dest='timeout',  default='3600', help="Second to wait between updates")

options = parser.parse_args()[0]
timeout = int(options.timeout)
user = options.username
print("User: %s" % user)

password = options.password
if not password: password = input()

PRP_FIRST_TIME = 150
PRP_DC = 151

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
    print("will sleep %d seconds" % timeout);
    sleep(timeout)
