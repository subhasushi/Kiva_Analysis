# Import dependencies

from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, func
from sqlalchemy import inspect

# import json to make json 
import json

# import mysql credentials

from credentials import *

# Create Engine and Pass in MySQL Connection
engine = create_engine(f"mysql://{user}:{password}@localhost:3306/kiva")

# prepare base
Base = automap_base()
Base.prepare(engine, reflect = True)

# create session
session = Session(engine)

# create reference to db tables
loans = Base.classes.loans
country = Base.classes.country
flags = Base.classes.flags

# run a query to get a unique list of country names
results = session.query(loans.country_name)
countries = []
for a in results:
    countries.append(a.country_name)
countries = set(countries)
clean = []
for a in countries:
    clean.append(a)

# create empty list. iterate through the list of countries
full_list = []
for country in clean:
    # start an empty dictionary
    empty = {"country" : country, "sectors" : [], "amounts" : []}
    # get country name, sector, total funded amount, limit to top 3, then add to dictionary
    results = session.query(loans.country_name, loans.sector_name, func.sum(loans.funded_amount_usd)).\
        filter(loans.country_name == country).\
        group_by(loans.country_name).\
        group_by(loans.sector_name).\
        order_by(func.sum(loans.funded_amount_usd).desc())
    for country,sector,loan in results[:3]:
        empty["sectors"].append(sector)
        empty["amounts"].append(int(loan))
    # get total loan count per country then add to dict
    total_loan_count_results = session.query(loans.country_name,func.count(loans.funded_amount_usd)).\
        filter(loans.country_name == country)
    for total_loan in total_loan_count_results:
        empty["total"] = total_loan[1]
    # get urls for each country then add to dict
    url_results = session.query(flags.country, flags.url).\
        filter(flags.country == country)
    for url in url_results:
        empty["url"] = url[1]
    full_list.append(empty)

# make json of final list

with open("/static/files/countries.json", w) as path:
    json.dump(full_list, path)

