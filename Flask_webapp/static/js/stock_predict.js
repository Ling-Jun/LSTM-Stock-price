$(document).ready(function() {
	$('form').on('submit', function(event) {
		// some text to decorate the button saying loading
		$.ajax({
			data : {
				// takes the html element with id=tickerInput s value
				ticker : $('#tickerInput').val(),
				// takes the html element with id=modelInput s value
				model : $('#modelInput').val()
				},
			type : 'POST',
			// 
			url : '/show_prediction'})
		.done(function(data) {
			// data.error, is the "error: " in app.py file's jsonify() return. 
			// In app.py file, show_pred() function has several returns, 
			// all the returns are in jsonify() form. 
			// The .ajax() function above takes the data as indicated above 
			// and outputs its jsonify() "error: " property to the html.
			if (data.error) {
				$('#errorMsg').text(data.error).show();
				// hide the html element with id=imgElem
				$("#imgElem").hide();
				// hide the html element with id=successAlert
				$('#successAlert').hide();
			}
			else {
				// using .html() method to output all parameters, html() reads string as html file
				$('#successAlert').html('<span style="font-weight:bold; color: blue;">Company Ticker: </span>'
																+ data.ticker +
																'<br><span style="font-weight:bold; color: blue;">Model: </span>'
																+ data.model +
																'<br><span style="font-weight:bold; color: blue;">RMSE: </span>'
																+ data.rmse + '<br><br>').show()
				// Reconstruct image from Base64 code to real IMGAGE by setting image attributes
				$("#imgElem").attr("src", "data:image/png;base64,"+ data.img).show();
				$('#errorMsg').hide();
			}
		});
		event.preventDefault();
	});
});
