function process_response(server_response){

    // check return response 
    var rc = server_response.rc;
    console.log(rc)
    
    if (rc==0) {
        
        $("#gallery_placeholder").empty()
        console.log(server_response.images)

        // append each of the images to gallery_placeholder
        for (var i = 0; i < server_response.images.length; i++) {
            
            // generic html
            var img_elem_to_append = 
            `<div class='gallery'> 
                <a target='_blank' href='image_location_placeholder'> 
                    <img src='image_location_placeholder' alt='Forest' width='300' height='200'>
                </a>
                <div class='desc'>image_description_placeholder</div>
            </div>`;

            // specify image location in generic html
            var img_elem_to_append = img_elem_to_append.replace("image_location_placeholder", server_response.images[i])
            var img_elem_to_append = img_elem_to_append.replace("image_location_placeholder", server_response.images[i])
            console.log(server_response.images[i])
            $("#gallery_placeholder").append(img_elem_to_append);
        }
        
    }
    else {

        // something went wrong

        $(".errors").empty();
        $(".errors").append("<p>Error. Server responded with rc : "+String(server_response.images)+"</p>")
    }

    // once query is over, enable searching
    $('#search_button').prop("disabled", false);
    $("#myquery").prop("disabled", false);

}

$("#search_button").click(function () {
            
            // vars
            var MIN_QUERY_LENGTH = 10;

            // Block user from searching anything else 
            $('#search_button').prop("disabled",true);
            $("#myquery").prop("disabled", true);

            // get the query
            query = String($("#myquery").val());
            console.log("user searched for: " + query);

            //basic checking 
            all_checks_ok = true;

            // check length > MIN_QUERY_LENGTH
            $(".errors").empty();
            if (query.length < MIN_QUERY_LENGTH) {
                $(".errors").append("<p>Err. Please enter a search query >" + String(MIN_QUERY_LENGTH) + " characters.</p>")
                all_checks_ok = false;
            }

            // pass query to server
            if (all_checks_ok == true){

                $.getJSON("/_process_query", { query: query}, process_response);

            }
            
        })