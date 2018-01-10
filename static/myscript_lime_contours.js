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

            // create result_div + canvas + append caption container div
            var $result_div = $("<div>", {"class": "gallery", "id": String(image_id)});

            var $result_div_img = $('<img class="true_image" src=' + server_response.flickr_urls[i] + 
                                        " alt='image' width=224 height=224> ").appendTo($result_div)

            var $result_div_canvas =  $('<canvas class="mycanvas" width=224 height=224></canvas>').appendTo($result_div)
            
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
        $("#gallery_placeholder").empty();
        $("#phrases").empty()
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

    // DISABLE_LIME_CONTOURS button
    var $elem = $('<p>'+'DISABLE_LIME_CONTOURS'+'</p>');
    $elem.css("float", "left")
    $elem.css("border", "2px solid red");
    $elem.css("padding", "10px");
    $elem.css("margin", "8px");
    $elem.css("font-size", "11px");
    $elem.css("border-radius", "25px");
    $phrases.append($elem);

    // buttons for all_phrases
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

function draw_lime_contours(image_id, contours_dict){
    
    // get the canvas 
    var $image_id_canvas = $('#'+String(image_id)).find("canvas")[0]; 
    var context = $image_id_canvas.getContext("2d");

    var cnt_idxs = Object.keys(contours_dict);
    cnt_idxs.forEach(function(cnt_idx){
        var contour = contours_dict[cnt_idx];
        context.beginPath()

        for(var i=0; i<contour.x.length; i++){
            if(i==0){
                context.moveTo(contour.x[i], contour.y[i]);
            }
            else{
                context.lineTo(contour.x[i], contour.y[i]);
            }
        }

        context.closePath();
        context.fillStyle="green";
        context.strokeStyle="green";
        context.lineWidth=3; 
        context.globalAlpha=1;
        context.stroke();
        context.globalAlpha=0.3;
        context.fill();

    });
    console.log("Drawn all contours for image_id"+image_id);
}

function show_salient_regions(){
    // This functions lights up the salient regions in all images corresponsing to clicked phrase 

    var phrase_clicked = $(this)[0].innerHTML; 
    
    // remove highlight background from every phrase
    for(var i=0; i<$("#phrases")[0].children.length; i++){
        $("#phrases")[0].children[i].style["background"] = "white";
    }

    // reset all canvases 
    var all_canvases = $(".mycanvas");
    all_canvases = all_canvases.toArray();
    all_canvases.forEach(function(canv){
        var context = canv.getContext("2d");
        context.clearRect(0, 0, canv.width, canv.height);
    });

    // if user clicked on DISABLE_CONTOURS button
    if (phrase_clicked=="DISABLE_LIME_CONTOURS"){
        return;
    }

    // highlight background for JUST THIS PHRASE
    $(this)[0].style["background"] = "lightgreen";

    // for each image Id -> draw the contours 
    Object.keys(LIME_RESULTS_OBJECT).forEach(function(image_id) {
        
        if (LIME_RESULTS_OBJECT[image_id][phrase_clicked] == null){
            return; //return if could not find any lime contour for (image_id, phrase_clicked)
        }

        var contours_dict = LIME_RESULTS_OBJECT[image_id][phrase_clicked];
        //debugger;
        draw_lime_contours(image_id, contours_dict); 
    });
    
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

var LIME_RESULTS_OBJECT = {} 
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
            LIME_RESULTS_OBJECT = {}

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

            // If devise-rnn ran correctly, there will be elements with class .true_image
            // Also, run LIME only for queries present in lime_queries.json 
            var availableTags = null; 
            $.getJSON("static/lime_queries.json", function(data){
                availableTags = data; 
            });
            // MAIN LIME IF CONDITION
            if ( ($('.true_image').length>0) && (availableTags.indexOf(query)>-1) ) {
            
            // LIME STARTS HERE
            //debugger;

            // 1. get phrases
            var all_phrases = null; 
            $.getJSON("/_get_phrases", { query: query}, function (data) {
                all_phrases = data.phrases
            });
            
            while(all_phrases==null){
                console.log("Trying to get all_phrases");
            }
            console.log(all_phrases);
            create_phrases(all_phrases); //append phrases to the phrase bar i.e div with id=phrases

            // 2. get all image_ids
            var image_ids = $(".gallery");
            image_ids = image_ids.toArray() //jquery returns an obj, need to convert it to an array for forEach 
            image_ids.forEach(function(im_id, index, theArray) {
                theArray[index] = im_id.id;
            });
            console.log(image_ids);

            // 3. Limit image_ids to 10 images 
            // (we return 50 images, but we have previously cached LIME for ONLY TOP 10 IMAGES)
            image_ids = image_ids.slice(0,10);

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

                // Empty dictionary for LIME_RESULTS_OBJECT[im_id]
                LIME_RESULTS_OBJECT[String(im_id)] = {} 

                // loop over every phrase
                //debugger;
                all_phrases.forEach(function(onePhrase){

                    // get explanation 
                    $.getJSON("/_get_LIME_contours", { phrase:JSON.stringify(onePhrase), image_id:JSON.stringify(im_id)}, function(response){

                        //debugger;
                        if(response["rc"] == 0){
                            
                            // cache contours in the LIME_RE.... object 
                            LIME_RESULTS_OBJECT[String(im_id)][String(onePhrase)] = response["lime"];

                            /*var $div = $("#"+String(im_id)); // div coresponding to that image_id
                            var overlay_img_elem_html = '<img class="some_class" src="some_src" width=224 height=224>'; 
                            overlay_img_elem_html = overlay_img_elem_html.replace("some_class", onePhrase);
                            overlay_img_elem_html = overlay_img_elem_html.replace("some_src", lime_image);
                            var $overlay_img = $(overlay_img_elem_html);
                            $div.prepend($overlay_img);*/
                            
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

            }); // end of image_ids.forEach , the ; is there to signify a statement 

            } // End of if($('.true_image')>0)
            
            
            
        }) // End of searchbutton click function call 
