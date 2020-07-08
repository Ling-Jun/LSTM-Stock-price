$(document).ready(function() {
	$('form').on('submit', function(event) {
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
				$('#successAlert').hide();
			}
			else {
				$('#successAlert').text(JSON.stringify(data)).show();
				$('#errorMsg').hide();
			}
		});
		event.preventDefault();
	});
});
