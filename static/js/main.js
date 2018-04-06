var map = L.map('map',{
    center:[0,0],
    zoomsnap: .25,
    zoom: 2.5,
    scrollWheelZoom: false
});
var streetmap = L.tileLayer("https://api.mapbox.com/styles/v1/mapbox/outdoors-v10/tiles/256/{z}/{x}/{y}?" +
"access_token=pk.eyJ1IjoibWluY2tpbTEyMjIiLCJhIjoiY2pkaGp5NHR0MHd3eDMxbnF6bXlsazhxYiJ9.6WOQPTje5_AYqQO_4W36xQ").addTo(map);

var dark = L.tileLayer("https://api.mapbox.com/styles/v1/mapbox/dark-v9/tiles/256/{z}/{x}/{y}?" +
"access_token=pk.eyJ1IjoibWluY2tpbTEyMjIiLCJhIjoiY2pkaGp5NHR0MHd3eDMxbnF6bXlsazhxYiJ9.6WOQPTje5_AYqQO_4W36xQ");

// add local json files to load the initial map, then iterate through to create geojson tooltips
var countryURL = "../static/files/countries.json"
var geoJsonURL = "../static/files/countries.geo.json"


function createGraph(country){
    var url = `/countries/${country}`;
    d3.json(url, function(error, response){
        if(error) console.warn(error);
        if(response.length != 0){
        }
    });
}

function getColor(d){
    return  d > 150000 ? '#800026' :
            d > 100000  ? '#bd0026' :
            d > 50000  ? '#e31a1c' :
            d > 30000  ? '#fc4e2a' :
            d > 20000  ? '#fd8d3c' :
            d > 10000   ? '#feb24c' :
                        '#ffeda0';
}

var legend = L.control({position: 'bottomright'});

legend.onAdd = function (map) {

    var div = L.DomUtil.create('div', 'info legend'),
        totals = [0, 10000, 20000, 30000, 50000, 100000, 150000],
        labels = [];

    // loop through our density intervals and generate a label with a colored square for each interval
    for (var i = 0; i < totals.length; i++) {
        div.innerHTML +=
            '<i style="background:' + getColor(totals[i] + 1) + '"></i> ' +
            totals[i] + (totals[i + 1] ? '&ndash;' + totals[i + 1] + '<br>' : '+');
    }

    return div;
};

legend.addTo(map);

var url = "../static/files/gender_disperity.json"
d3.json(url, function(error, response) {
    
  // console.log(response);
 
  var femaleMarkers = [];
  var maleMarkers = [];

  //Parse through json object to extract pop-up data
   for (var i = 0; i < response.length; i++) {

    femaleMarkers.push(
    L.minichart([response[i].LATITUDE, response[i].LONGITUDE], {type:'pie',data: [response[i].FEMALE, response[i].MALE],labels:'auto',
    labelMinSize:8, labelMaxSize:10,width:40,height:40})
    .bindPopup("<h5>" + response[i].COUNTRY + "</h5>\
    <ul class=list-group>\
    <li class=list-group-item>" + "Female_count: " + response[i].FEMALE + "</li>\
    <li class=list-group-item>" + "Male_count: " + response[i].MALE + "</li>"
  ))

  }  
  
  var female = L.layerGroup(femaleMarkers);
  d3.json(countryURL, function(error, response){
    d3.json(geoJsonURL, function(error, geoResponse){
        var geoLayer = L.geoJson(geoResponse,{
            style: function(feature) {
                if(feature.properties.total)
                {
                return{
                fillColor: getColor(feature.properties.total),
                weight: 2,
                opacity: .5,
                dashArray: '3',
                fillOpacity: 0.5
                };
                } else {
                    return{
                        fillOpacity: .5
                    }
                }
            },
            onEachFeature: function(feature, layer) {
                for (i = 0; i <response.length; i++){
                    var curCountry = response[i];
                    var countryName = curCountry.country;
                    var topSector = curCountry.sectors[0];
                    var secSector = curCountry.sectors[1];
                    var thirdSector = curCountry.sectors[2];
                    var topAmount = curCountry.amounts[0];
                    var secAmount = curCountry.amounts[1];
                    var thirdAmount = curCountry.amounts[2];
                    var totLoans = curCountry.total;
                    if(countryName == feature.properties.name){
                        layer.bindPopup(`<h4>${countryName} </h4>
                        <table>
                            <tr><th>Sector</th><th align="right">Amount Funded</th><tr>
                            <tr><td>${topSector}</td><td align="right">$${topAmount}</td>
                            <tr><td>${secSector}</td><td align="right">$${secAmount}</td>
                            <tr><td>${thirdSector}</td><td align="right">$${thirdAmount}</td>
                        </table>
                        <h4>Total loans : ${totLoans}</h4>`)
                    } 
                }
                layer.on({
                // When a user's mouse touches a map feature, the mouseover event calls this function, that feature's opacity changes to 90% so that it stands out
                mouseover: function(event) {
                    layer = event.target;
                    layer.setStyle({
                    fillOpacity: 0.8
                    });
                },
                // When the cursor no longer hovers over a map feature - when the mouseout event occurs - the feature's opacity reverts back to 50%
                mouseout: function(event) {
                    layer = event.target;
                    layer.setStyle({
                    weight: 2,
                    opacity: .5,
                    dashArray: '3',
                    fillOpacity: 0.5
                    });
                },
                // When a feature (neighborhood) is clicked, it is enlarged to fit the screen
                click: function(event) {
                    map.fitBounds(event.target.getBounds());
                    var currentClickedCountry = event.target.feature.properties.name
                    createGraph(currentClickedCountry);
                }
                });                
            }
        })
        var overlayMaps = {
            "Total Loan Choropleth": geoLayer,
            "Gender Pie Chart" : female
        };

        var baseMaps = {
            "Street" : streetmap,
            "Dark" : dark
        }
        L.control.layers(baseMaps, overlayMaps).addTo(map); 
        function hide_charts(e) {
            e.layer.eachLayer(
               function(t) {
                  if (t._chart) { t._chart.remove(); }
               }
            );
         }
         map.on('overlayremove', hide_charts)
    });
   
})
});

// var stackedURL = "../static/files/stacked_by_year.json"
// d3.json(stackedURL, function(error, response){
//     console.log(response);
//     }
// );
var url = "/gender_growth_over_years";
femaleBorrowers = []
year = []
maleBorrowers = []

var countryList = []  
var sectorNames = [];  
var femaleCount = []; 
var maleCount = []     

var sectoreResponseData
var borrowerResponseData

function getCountryData(chosenCountry) {

    var country_yearwise_list = borrowerResponseData[chosenCountry]
    femaleBorrowers = []
    year = []
    maleBorrowers = []
    for (var i = 0 ; i < country_yearwise_list.length ; i++){
        
            countryData = country_yearwise_list[i]                    
            femaleBorrowers.push(countryData.FEMALE)
            maleBorrowers.push(countryData.MALE)
            year.push(countryData.YEAR)                    
    }     
}    

function setBubblePlot(chosenCountry) {
    getCountryData(chosenCountry);        

    var trace1 = {
        x:year ,
        y: femaleBorrowers,
        mode: 'lines+markers',
        name: "Female Borrower Growth Over Years",
        marker: {
            size: 12,
            opacity: 0.5
        },
        line: {
            color: "Green"
        }
    };

    var trace2 = {
        x: year,
        y: maleBorrowers,
        mode: 'lines+markers',
        name: "Male Borrower Growth Over Years",
        marker: {
            size: 12,
            opacity: 0.5
        },
        line: {
            color: "Red"
        }
    };



    var data = [trace1, trace2];

    var layout = {
        title: "Borrower Gender Growth Over Years",
        xaxis: {
            type: "date"
        },
        yaxis: {
            autorange: true,
            type: "linear"
        }
    };

    Plotly.newPlot("plot", data, layout);
}


Plotly.d3.json(url, function(error, response) {

        console.log(response);
        borrowerResponseData = response    
        
        for (countryName in response){       

            var each = response[countryName]         
            countryList.push(countryName)
        }
        
        assignOptions(countryList, countrySelector);
        setBubblePlot('Afghanistan');
});

// // Default Country Data      


// // // /* data route */

function getSectorCountryData(chosenCountry) {

    var country_yearwise_list = sectoreResponseData[chosenCountry]
    sectorNames = [];  
    femaleCount = []; 
    maleCount = []       
    
    for (var i = 0 ; i < country_yearwise_list.length ; i++){
        
            countryData = country_yearwise_list[i]
            // console.log(countryData)                    
            femaleCount.push(countryData.Female)
            maleCount.push(countryData.Male)
            sectorNames.push(countryData.Sector)                    
    }     
} 

function setBarPlot(chosenCountry) {
    getSectorCountryData(chosenCountry);        

    var trace1 = {
        x: sectorNames,
        y: femaleCount,
        type: 'bar',
        name: "Female Count By Sector",
        
    };

    var trace2 = {
        x: sectorNames,
        y: maleCount,
        type: 'bar',
        name: "Male Count By Sector",
    };


    var data = [trace1, trace2];

    var layout = {
        title: "Sector Popularity by Gender",
        barmode: 'group'
    };

    Plotly.newPlot("sectorplot", data, layout);
}
var url2 = "/genderwise_popular_sector";

Plotly.d3.json(url2, function(error, response) {

    sectoreResponseData = response
    // for (countryName in response){
    //     var each = response[countryName]         
    //     countryList.push(countryName)
    // } 
    
    // // Default Country Data      
    setBarPlot('Afghanistan');  
});

var innerContainer = document.querySelector('[data-num="0"'),
    plotEl = innerContainer.querySelector('#plot', '#sectorplot'),
    countrySelector = innerContainer.querySelector('.countrydata');

function assignOptions(textArray, selector) {
    for (var i = 0; i < textArray.length;  i++) {
        var currentOption = document.createElement('option');
        currentOption.text = textArray[i];
        selector.appendChild(currentOption);
    }
}    

function updateCountry(){
    setBubblePlot(countrySelector.value);
    setBarPlot(countrySelector.value);
}

countrySelector.addEventListener('change', updateCountry, false);


