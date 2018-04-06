# Import Splinter and BeautifulSoup
from credentials import *
from splinter import Browser
from bs4 import BeautifulSoup

# create sql object to connect to db to get country list
from sqlalchemy import create_engine
from sqlalchemy.ext.automap import automap_base

# random dependencies
import urllib
import time, random

# Create Engine and Pass in MySQL Connection
engine = create_engine(f"mysql://{user}:{password}@localhost:3306/kiva")

# create browser object
browser = Browser('chrome', headless=True)

# create search url
init_url = "http://flagpedia.net/s?q="

# connect to db
# create base
Base = automap_base()
Base.prepare(engine, reflect = True)
# set sqlalchemy tables and conn
conn = engine.connect()
loans = Base.classes.loans
country = Base.classes.country

#query for all countries with loans, set it to get unique values, then reappend to a clean list
results = session.query(loans.country_name)
countries = []
for a in results:
    countries.append(a.country_name)
clean_country_list = []
countries = set(countries)
for country in countries:
    clean_country_list.append(country)
# this iterates through the country list, visits the website and retrieves the url for the flag, then saves it to a list of dictionaries
# first remove two countries that caused problems
clean_country_list.remove("Palestine", "Timor-Leste")

# empty list to hold data
flag_list = []
for country in clean_country_list:
    flag_dict = {}
    url = init_url + country
    browser.visit(url)
    html = browser.html
    soup = BeautifulSoup(html, 'html.parser')
    initial = soup.find("div", id="container").find("h1")
    if initial.text == "Search":
        browser.find_by_css("td.td-flag").click()
        html = browser.html
        soup = BeautifulSoup(html, "html.parser")
        cur_url = soup.find("p", id="flag-detail").find("img")["src"]
        full_url = "http:" + cur_url        
    else:
        html = browser.html
        soup = BeautifulSoup(html, "html.parser")
        cur_url = soup.find("p", id="flag-detail").find("img")["src"]
        full_url = "http:" + cur_url
    flag_dict["Country"] = country
    flag_dict["Image URL"] = full_url
    flag_list.append(flag_dict)
    # random sleep to make it less bot-like
    time.sleep(random.randint(20,40))

# manually get the 2 problematic countries from earlier and append them

blar = {}
blar["Country"] = "Timor-Leste"
blar["Image URL"] = "http://flags.fmcdn.net/data/flags/w580/tl.png"
flag_list.append(blar)

blar2 = {}
blar2["Country"] = "Palestine"
blar2["Image URL"] = "https://en.wikipedia.org/wiki/Flag_of_Palestine#/media/File:Flag_of_Palestine.svg"
flag_list.append(blar2)

# change completed list to pandas dataframe to save as csv file to add to main kiva db
import pandas as pd 

flag_df = pd.DataFrame(flag_list)
flag_df = flag_df.set_index("Country")
flag_df.to_csv("flags.csv")
