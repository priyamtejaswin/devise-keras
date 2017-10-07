function process_response(server_response){

    // check return response 
    var rc = server_response.rc;
    //console.log(rc);
    
    if (rc==0) {
        
        $("#gallery_placeholder").empty();
        $("#phrases").empty()
        //debugger;
        //console.log(server_response.images);

        // append each of the images to gallery_placeholder
        for (var i = 0; i < server_response.coco_urls.length; i++) {
            
            // get image id 
            var image_id = server_response.coco_urls[i]; //mscoco.org/32561
            image_id = image_id.split("/")[image_id.split("/").length-1] // get the number 32561
            image_id = image_id.replace(".jpg", "");

            // create div element + append image + append caption container div
            var $result_div = $("<div>", {"class": "gallery", "id": String(image_id)});
            var $result_div_img = $('<img class="true_image" src=' + server_response.flickr_urls[i] + " alt='image' width=245 height=150> ").appendTo($result_div)
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
        $("#errors").append("<p>Error. Server responded with rc : "+String(server_response.flickr_urls)+"</p>")
    }

    // enable the search bar
    $('#search_button').prop("disabled", false);
    $("#myquery").prop("disabled", false);

}

function create_phrases(all_phrases) {
    // This function creates clickable boxes for each phrase split from user's query 
    // Clicking on any of the box should show the LIME result for that phrase on all images
    
    var $phrases = $("#phrases");
    for (var k = 0; k < all_phrases.length; k++){

        var $elem = $('<p>'+all_phrases[k]+'</p>');

        $elem.css("float", "left")
        $elem.css("border", "2px solid #73AD21");
        $elem.css("padding", "10px");
        $elem.css("margin", "8px");
        $elem.css("font-size", "11px");

        //$elem.css("height", "40px");
        $elem.css("border-radius", "25px");

        $phrases.append($elem); 
    }

    console.log("appended phrases to phrase bar");
}

function show_salient_regions(){
    // This functions lights up the salient regions in all images corresponsing to clicked phrase 

    //debugger;
    
    var phrase_clicked = $(this)[0].innerHTML;
    var all_phrases    = $("#phrases").children()
    var all_phrases_clean = [];
    for (var i=0; i < all_phrases.length; i++){
        all_phrases_clean[i] = all_phrases[i].innerHTML;
    }
    all_phrases = all_phrases_clean;

    // make true_image opacity = 0.8 
    $(".true_image").css("opacity", 0.3);

    // make all phrase image opacity = 0.0
    for (var i=0; i < all_phrases.length; i++){
        $("."+all_phrases[i]).css("opacity",0.0);
    }

    // make phrase_clicked opacity = 0.5
    $("."+phrase_clicked).css("opacity",0.95); 

}

function split( val ) {
    return val.split( /,\s*/ );
}
function extractLast( term ) {
    return split( term ).pop();
}

$( function() {
    var availableTags = (function() {
        var availableTags = null;
        $.ajax({
            'async': false,
            'global': false,
            'url': "static/lime_queries.json",
            'dataType': "json",
            'mimeType': 'application/json',
            'type':        "GET",
            'success': function (data) {
                availableTags = data;
            }
        });
        return availableTags;
    })();
    console.log(availableTags);

    function split( val ) {
      return val.split( /,\s*/ );
    }
    function extractLast( term ) {
      return split( term ).pop();
    }

    $( "#myquery" )
      // don't navigate away from the field on tab when selecting an item
      .on( "keydown", function( event ) {
        if ( event.keyCode === $.ui.keyCode.TAB &&
            $( this ).autocomplete( "instance" ).menu.active ) {
          event.preventDefault();
        }
      })
      .autocomplete({
        minLength: 0,
        source: function( request, response ) {
          // delegate back to autocomplete, but extract the last term
          response( $.ui.autocomplete.filter(
            availableTags, extractLast( request.term ) ) );
        },
        focus: function() {
          // prevent value inserted on focus
          return false;
        },
        select: function( event, ui ) {
          var terms = split( this.value );
          // remove the current input
          terms.pop();
          // add the selected item
          terms.push( ui.item.value );
          // add placeholder to get the comma-and-space at the end
          terms.push( "" );
          this.value = terms.join( "" );
          return false;
        }
      });
} );

$("#search_button").click(function () {

            // All Ajax are synchronous because we want to do things in order
            $.ajaxSetup({
                async: false
            }); 
            
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

            // LIME STARTS HERE

            // 1. get phrases
            var all_phrases = null; 
            $.getJSON("/_get_phrases", { query: query}, function (data) {
                all_phrases = data.phrases
            });
            
            while(all_phrases==null){
                console.log("Trying to get all_phrases");
            }
            console.log(all_phrases)
            create_phrases(all_phrases) //append phrases to the phrase bar i.e div with id=phrases

            // 2. get all image_ids
            var image_ids = $(".gallery");
            image_ids = image_ids.toArray() //jquery returns an obj, need to convert it to an array for forEach 
            image_ids.forEach(function(im_id, index, theArray) {
                theArray[index] = im_id.id;
            });

            console.log(image_ids);

            // 3. Make the phrase tags clickable and do something with it 
            var phrase_elems = $("#phrases").children().click(show_salient_regions);

            // 4. run lime for each image
            image_ids.forEach(function(im_id, index, thearray){

                //for each image 

                // show in ui that explanation is being loaded
                var error_bar = $("#errors");
                error_bar.empty();
                var explanation_load = $("<p>Loading Explanation for Image ID :" + String(im_id) + " </p>");
                error_bar.append(explanation_load);

                // loop over every phrase
                debugger;
                all_phrases.forEach(function(onePhrase){

                    // get explanation 
                    $.getJSON("/_get_LIME", { phrase:JSON.stringify(onePhrase), image_id:JSON.stringify(im_id)}, function(response){

                        //debugger;
                        if(response["rc"] == 0){
                            // phrase_imgs for im_id
                            lime_image = response["lime"];

                            var $div = $("#"+String(im_id)); // div coresponding to that image_id
                            var overlay_img_elem_html = '<img class="some_class" src="some_src" width=245 height=150>'; 
                            overlay_img_elem_html = overlay_img_elem_html.replace("some_class", onePhrase);
                            overlay_img_elem_html = overlay_img_elem_html.replace("some_src", lime_image);
                            var $overlay_img = $(overlay_img_elem_html);
                            $div.prepend($overlay_img);
                            
                        }
                        else{
                            console.log("something went wrong for image: " + im_id + " and phrase: " + onePhrase);
                        }
                    });
                });

                // show in ui if all explanations have been loaded 
                if(index == thearray.length-1) {
                    error_bar.empty();
                    error_bar.append($("<p>All explanations Loaded!</p>"));
                }

            }); 
            
            
        })
