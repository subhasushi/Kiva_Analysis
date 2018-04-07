var column_name = {
    'female_count'       :'Number of female borrowers',
    'male_count'         :'Number of male borrowers',
    'lender_term'        :'Loan Term (in months)',
    'num_lenders_total'  :'Number of lenders',
    'loan_amount'        :'Loan Amount($)',
    'posted_raised_hours':'Hours to raise full amount',
    'population_in_mpi'  :'Population % in MPI',
    'num_journal_entries':'Number of journal entries',
    'activity_name'      :'Activity',
    'sector_name'        :'Sector',
    'country'            :'Country',
    'status'             :'Status'
};

function populateCluster(n){

    cluster_title = d3.select('#cluster-title').attr('class','text-center').style('padding','14px');
    cluster_title.html('');

    cluster_title.append('h3')
            .text(`Predicted Cluster - ${n}`);

    d3.json(`/clusters/${n}`).then(function(data){
        console.log('Got back something!!!');
        console.log(data['numerical'][0]);

        numerical_section = d3.select('#numerical-statistics');

        numerical_section.html('');

        numerical_data = numerical_section.append('table').attr('class','table table-hover font-weight-bold').attr('border','1px');

        numerical_data.selectAll('tbody')
                .data(data['numerical'])
                .enter()
                .append('tbody')
                .html(function(d){   

                    console.log('asdas');
                    var inner_data = '';
                    var keys = Object.keys(d);

                    console.log(keys);
                    
                    for (var i in keys){
                        console.log(i);
                        inner_data = inner_data+`<tr><td> ${column_name[keys[i]]} </td> <td> ${d[keys[i]]} </td></tr>`;
                    };

                    console.log(inner_data);

                    return inner_data;
                });

        console.log(data['numerical']);

        var top_n_country = d3.select('#top-n-country');
        var top_n_sector = d3.select('#top-n-sector');
        var top_n_activity = d3.select('#top-n-activity');
        var top_n_status = d3.select('#top-n-status');

        top_n_country.html('');
        top_n_sector.html('');
        top_n_activity.html('');
        top_n_status.html('');

        top_n_country.append('h3').classed("text-center",true).text('Countries');
        top_n_sector.append('h3').classed("text-center",true).text('Sectors');
        top_n_activity.append('h3').classed("text-center",true).text('Activities');
        top_n_status.append('h3').classed("text-center",true).text('Status');

        activity_keys = Object.keys(data['categorical'][0]['activity_name'][0]);
        activities = data['categorical'][0]['activity_name'][0];

        top_n_activity.selectAll('div')
                .data(activity_keys)
                .enter()
                .append('div')
                
                .style('font-size','14px')
                .text(function(d){
                    console.log("try",activities[d]);
                    return `${d} - ${activities[d]}`;
                });
        

        sector_keys = Object.keys(data['categorical'][0]['sector_name'][0]);
        sectors = data['categorical'][0]['sector_name'][0];

        top_n_sector.selectAll('div')
                .data(sector_keys)
                .enter()
                .append('div')
                
                .style('font-size','14px')
                .text(function(d){
                    console.log("try",sectors[d]);
                    return `${d} - ${sectors[d]}`;
                });

        country_keys = Object.keys(data['categorical'][0]['country'][0]);
        countries = data['categorical'][0]['country'][0];

        top_n_country.selectAll('div')
                .data(country_keys)
                .enter()
                .append('div')
                
                .style('font-size','14px')
                .text(function(d){
                    console.log("try",countries[d]);
                    return `${d} - ${countries[d]}`;
                });

        status_keys = Object.keys(data['categorical'][0]['status'][0]);
        status = data['categorical'][0]['status'][0];

        top_n_status.selectAll('div')
                .data(status_keys)
                .enter()
                .append('div')
                
                .style('font-size','14px')
                .text(function(d){
                    console.log("try",status[d]);
                    return `${d}`;
                });
    });
}

function init(){
    d3.json('/predictNewLoans').then(function(data){
        console.log(data);

    var cols = Object.keys(data[0]);
    console.log(cols);


    var tbody = d3.select('#sample_tbody').style('font-size','14px');

    tbody.html('');

    console.log('Sample tbody',tbody);

    var rows = tbody.selectAll('tr')
                    .data(data)
                    .enter()
                    .append('tr')
                    .on('click',function(d,i){
                        d3.select('#selected-sample').html(`<strong> Selected sample # ${i} </strong><br>`);
                        console.log('Clicked',d);
                        populateCluster(d['class']);
                    });

    var cells = rows.selectAll('td')
                    .data(function(d) {
                        
                        console.log(cols);
                        return cols.map(function(feature) {
                            if(feature == 'class')
                                return ""
                            return d[feature];
                        })
                    })
                    .enter()
                    .append('td')
                    .text(d=>d)
                    ; 
    });

}

init()
