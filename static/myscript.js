 $("#search_button").click(function () {
            
            // vars
            var MIN_QUERY_LENGTH = 10;

            // get the query
            query = String($("#myquery").val());
            console.log(query);

            //basic checking 
            all_checks_ok = true;

            // check length > MIN_QUERY_LENGTH
            $(".errors").empty();
            if (query.length < MIN_QUERY_LENGTH) {
                $(".errors").append("<p>Err. Please enter a search query >" + String(MIN_QUERY_LENGTH) + " characters.</p>")
                all_checks_ok = false;
            }

            if (all_checks_ok == true){

                $.getJSON("/_process_query", 
                {
                    a: 5,
                    b: 6
                },
                function (data){console.log("The result is "+data.result)}
            )    
            }
            
            
        })