$(document).ready(function() {
	// takes data from the HTML 'form, there's ONLY ONE form in the HTML in our case,
	$('form').on('submit', function(event) {
		// some text to decorate the button saying loading
		$.ajax({
			// is "data" an AJAX keyword?
			data : {
				// # is followed by the element ID in html
				ticker : $('#tickerInput').val(),
				model : $('#modelInput').val()
				},
			type : 'POST',
			// POSTs the data to "/show_prediction" endpoint in app.py to run the ML model
			url : '/show_prediction'})
			//.then(function()){}
			//when the process is DONE through endpoint/backend "/show_prediction", execute the function in .done()
		.done(function(data) {
			if (data.error) {
				$('#errorMsg').text(data.error).show();
				$("#imgElem").hide();
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
