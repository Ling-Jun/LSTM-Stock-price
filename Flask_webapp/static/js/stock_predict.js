$(document).ready(function() {
	$('form').on('submit', function(event) {
		// some text to decorate the button saying loading
		$.ajax({
			data : {
				ticker : $('#tickerInput').val(),
				model : $('#modelInput').val()
				},
			type : 'POST',
			url : '/show_prediction'})
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
