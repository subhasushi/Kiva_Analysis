# import necessary libraries
# import pandas as pd
import os
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, func, inspect
import mytestutilities as myutilities
import pickle

import pandas as pd

# PyMySQL 
import pymysql
pymysql.install_as_MySQLdb()
# Create Engine and Pass in MySQL Connection

db_url = os.environ['CLEARDB_DATABASE_URL']
engine = create_engine(db_url, pool_recycle=300, pool_pre_ping=True)

# engine = create_engine(f"mysql://root:root@localhost:3306/kiva")

#################################################
# Database Setup
#################################################


# Create Base
Base = automap_base()
Base.prepare(engine, reflect = True)

# Create our session (link) from Python to the DB
session = Session(engine)
conn = engine.connect()

# Set tables

loans = Base.classes.loans
country = Base.classes.country
# flags = Base.classes.flags

# Create empty list to get column names for querying

column_list_loans = []
column_list_country = []
for column in loans.__table__.columns:
    column_list_loans.append(column.key)
for column in country.__table__.columns:
    column_list_country.append(column.key)

# print(column_list_country)
# print(column_list_loans)


from flask import (
    Flask,
    render_template,
    jsonify)


#################################################
# Flask Setup
#################################################
app = Flask(__name__)

@app.teardown_request
def session_clear(exception=None):
    if exception and session.is_active:
        session.rollback()

# create route that renders index.html template
@app.route("/")
def home():
    return render_template("index.html")

# route queries and returns a list of the country names. will be used to populate dropdown
@app.route("/countries")
def get_countries():
    results = session.query(loans.country_name)
    countries = []
    for a in results:
        countries.append(a.country_name)
    countries = set(countries)
    clean = []
    for a in countries:
        clean.append(a)
    
    return jsonify(clean)

# Get country info
@app.route("/countries/<country>")
def get_country_info(country):
    country = country.capitalize()
    results = session.query(loans).\
        filter(loans.country_name == country).all()
    full_dict = []
    for result in results:
        current_dict = {}
        current_dict["name"] = result.country_name
        current_dict["country_code"] = result.country_code
        current_dict["loan_id"] = result.loan_id
        current_dict["status"] = result.status
        current_dict["sector"] = result.sector_name
        current_dict["activity"] = result.activity_name
        current_dict["posted"] = result.posted_time
        current_dict["disbursed"] = result.disburse_time
        current_dict["raised"] = result.raised_time
        current_dict["term"] = result.lender_term
        current_dict["total_lenders"] = result.num_lenders_total
        current_dict["interval"] = result.repayment_interval
        current_dict["female"] = result.female_count
        current_dict["male"] = result.male_count
        current_dict["borrowers"] = result.borrower_count
        current_dict["funded_amount"] = result.funded_amount_usd
        current_dict["loan_amount"] = result.loan_amount_usd
        current_dict["shortage_amount"] = result.shortage_fund
        full_dict.append(current_dict)
    
    return jsonify(full_dict)

# Route to get flag url
@app.route("/flag/<country>")
def get_country_flag(country):
    country = country.capitalize()
    flag_query = session.query(flags).\
        filter(flags.country == country)
    for flag in flag_query:
        flag_dict = {}
        flag_dict["country"] = flag.country
        flag_dict["url"] = flag.url

    return jsonify(flag_dict)

@app.route("/gender_disperity")
def gender_count():
    results = session.query(
    func.sum(loans.female_count).label('female_count'), func.sum(loans.male_count).label('male_count'), loans.country_name,
    country.latitude,country.longitude
    ).join(country, loans.country_code==country.country_code
    ).group_by(loans.country_name
    ).all()

    gender_count_country = []

    # Create a dictionary entry for each row of metadata information
    
    for result in results:
        gender_metadata = {}
        gender_metadata['COUNTRY'] = result[2]
        gender_metadata['FEMALE'] = int(result[0])
        gender_metadata['MALE'] = int(result[1])
        gender_metadata['LATITUDE'] = int(result[3])
        gender_metadata['LONGITUDE'] = int(result[4])
        gender_count_country.append(gender_metadata)
        
    return jsonify(gender_count_country)

@app.route("/stacked_by_year")
def get_stacked_bar():

    results = session.query(loans.sector_name, func.extract("year", loans.posted_time).label("year"), func.sum(loans.funded_amount_usd)).\
        group_by(loans.sector_name).\
        group_by("year")
    empty = []
    for a in results:
        l = {}
        l["sector"] = a[0]
        l["year"] = a[1]
        l["amount"] = int(a[2])
        empty.append(l)
    return jsonify(empty)
    
@app.route("/genderwise_popular_sector")
def sector_popularity():
    
    sel = [func.sum(loans.female_count).label('female_borrower_count'), func.sum(loans.male_count).label('male_borrower_count'), loans.sector_name,loans.country_name]
    results = session.query(*sel).\
        group_by(loans.country_name,loans.sector_name).all()

    # print(results)

    master_list_sector = {}

    for result in results:
        sector_data = {}
        sector_data['Female'] = int(result[0])
        sector_data['Male'] = int(result[1])
        sector_data['Sector'] = (result[2])

        if  result[3] in master_list_sector:
            master_list_sector
            # print ("result[3]" + result[3])
            master_list_sector[result[3]].append(sector_data)
        else :
            
            master_list_sector[result[3]] = [sector_data]
        

    return jsonify(master_list_sector)

@app.route("/topCountries")
def topCountries():
    sel = [func.count(loans.loan_id), loans.country_name]
    results = session.query(*sel).\
        group_by(loans.country_name).order_by(func.count(loans.loan_id).desc()).all()

    # print(results)
    top_3_Countries = []
    for result in results:
        country_details = {}
        country_details["Country"] = (result[1])
        country_details["NumberOfLoans"] = int(result[0])
        top_3_Countries.append(country_details)

    return jsonify(top_3_Countries)

@app.route("/topCountriesByLoanCount")
def topCountry():
    return render_template("topCountry.html")


@app.route("/philippinesData")
def philippinesData():
    sel = session.query(loans.sector_name, loans.activity_name, loans.loan_amount_usd, loans.country_name, func.count(loans.loan_id), loans.num_lenders_total).filter(loans.country_name == "Philippines").\
    group_by(loans.sector_name).all()
   
    return jsonify(sel)



@app.route("/kenyaData")
def kenyaData():
    sel = session.query(loans.sector_name, loans.activity_name, loans.loan_amount_usd, loans.country_name, func.count(loans.loan_id), loans.num_lenders_total).filter(loans.country_name == "Kenya").\
    group_by(loans.sector_name).all()
   
    return jsonify(sel)



@app.route("/peruData")
def peruData():
    sel = session.query(loans.sector_name, loans.activity_name, loans.loan_amount_usd, loans.country_name, func.count(loans.loan_id), loans.num_lenders_total).filter(loans.country_name == "Peru").\
    group_by(loans.sector_name).all()
   
    return jsonify(sel)   

@app.route("/gender_growth_over_years")
def gender_count2():
    # year_list = []
    # year = session.query(func.extract('year', Loans.posted_time).distinct())
    # for years in year:
    #     year_list.append(years)
    # print(year_list)
    sel = [func.sum(loans.female_count).label('female_borrower_count'), func.sum(loans.male_count).label('male_borrower_count'), func.extract("year", loans.posted_time), loans.country_name]
    results = session.query(*sel).\
        group_by(loans.country_name,(func.extract("year", loans.posted_time))).all()

    # print(results)   
    

    master_list = {}

    for result in results:
        gender_yeardata = {}
        
        gender_yeardata['YEAR'] = result[2]
        gender_yeardata['FEMALE'] = int(result[0])
        gender_yeardata['MALE'] = int(result[1])
        #gender_count_year.append(gender_yeardata)

        if  result[3]  in master_list:
            master_list
            # print ("result[3]" + result[3])
            master_list[result[3]].append(gender_yeardata)
        else :
            
            master_list[result[3]] = [gender_yeardata]
            # print ( master_list )


    return jsonify(master_list)


@app.route("/view_clusters")
def clusters():
    return render_template("clusters.html")

@app.route("/clusters")
def getClusters():   
    loan_samples_clustered = pd.read_csv('Clustered_samples_dataset.csv')

    category_data = myutilities.describe_categorical(loan_samples_clustered)
    numerical_data = myutilities.describe_numerical(loan_samples_clustered)

    data = {
        'categorical':category_data,
        'numerical':numerical_data
    }

    return jsonify(data)

@app.route("/clusters/<n>")
def getClusterData(n):   
    loan_samples_clustered = pd.read_csv('Clustered_samples_dataset.csv')

    category_data = [myutilities.single_cluster_categorical(loan_samples_clustered,n)]
    numerical_data = [myutilities.single_cluster_numerical(loan_samples_clustered,n)]

    data = {
        'categorical':category_data,
        'numerical':numerical_data
    }

    return jsonify(data)

@app.route("/predictNewLoans")
def getNewLoanPrediction():   
    latest_loans = myutilities.getLatestLoans()

    print('Latest Loans',len(latest_loans))

    predicted_df = pd.DataFrame()

    if(len(latest_loans)>0):
        feature_df = myutilities.getFeatureVector(latest_loans)

        print(len(feature_df))

        transformed_df = myutilities.applyPCA(3,feature_df)

        predicted_df = latest_loans

        pkl_filename = "Kiva_4_Clusters.pkl"  

        # Load from file
        with open(pkl_filename, 'rb') as file:  
            cluster_model = pickle.load(file)
            print('pickiling')

        predicted_df['class'] = cluster_model.predict(transformed_df)
        print('predicted')

        print(len(predicted_df))

        samples = predicted_df
        samples = samples.drop_duplicates()
        
        samples = samples[:10][["country", "loan_amount","sector_name","activity_name", "borrower_count","class"]].to_dict(orient='records')
    return jsonify(samples)

if __name__ == "__main__":
    app.run(debug=True)

