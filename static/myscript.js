function process_response(server_response){

    // check return response 
    var rc = server_response.rc;
    //console.log(rc);
    
    if (rc==0) {
        
        $("#gallery_placeholder").empty();
        //console.log(server_response.images);

        // append each of the images to gallery_placeholder
        for (var i = 0; i < server_response.images.length; i++) {
            
            // generic html
            var img_elem_to_append = 
            `<div class='gallery'> 
                <img src='image_location_placeholder' alt='image' width=245 height=150>
                <img id="overlay" src='image_location_placeholder_2' alt='image' width=245 height=150>
                <div id='true_captions_xx'>
                   
                </div>
            </div>`;

            // get image id 
            var image_id = server_response.images[i]; //mscoco.org/32561
            image_id = image_id.split("/")[image_id.split("/").length-1] // get the number 32561

            // create div element + append image + append caption container div
            var $result_div = $("<div>", {"class": "gallery", "id": String(image_id)});
            var $result_div_img = $('<img src=' + server_response.images[i] + " alt='image' width=245 height=150>").appendTo($result_div)
            var $result_div_captions = $('<div id="true_captions"></div>').appendTo($result_div)

            // add captions to $result_div_captions 
            for (var k = 0; k < server_response.captions[i].length; k++) {
                $result_div_captions.append("<p>"+server_response.captions[i][k]+"</p>"); // append the caption in the correct image ; refer using true_caption_{integer value}
            }

            // put it in the DOM
            $("#gallery_placeholder").append($result_div);

        }
        
    }
    else {

        // something went wrong
        $("#errors").empty();
        //console.log(server_response.images);
        $("#errors").append("<p>Error. Server responded with rc : "+String(server_response.images)+"</p>")
    }

    // enable the search bar
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
            $("#errors").empty();
            if (query.length < MIN_QUERY_LENGTH) {
                $("#errors").append("<p>Err. Please enter a search query >" + String(MIN_QUERY_LENGTH) + " characters.</p>")
                all_checks_ok = false;
            }

            // pass query to server
            if (all_checks_ok == true){

                $.getJSON("/_process_query", { query: query}, process_response);

            }

            // enable the search bar 
            $('#search_button').prop("disabled", false);
            $("#myquery").prop("disabled", false);
            
        })
