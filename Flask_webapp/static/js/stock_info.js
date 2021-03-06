$(document).ready(function() {
	$('form').on('submit', function(event) {
		$.ajax({
			data : {
				ticker : $('#tickerInput').val()
				},
			type : 'POST',
			url : '/process'})
		.done(function(data) {
			if (data.error) {
				$('#errorAlert').text(data.error).show();
				$('#successAlert').hide();
			}
			else {
				// $('#successAlert').text(JSON.stringify(data)).show();
				// html() function reads html raw code and create html page
				$('#successAlert').html(data).show();
				$('#errorAlert').hide();
			}
		});
		event.preventDefault();
	});
});
