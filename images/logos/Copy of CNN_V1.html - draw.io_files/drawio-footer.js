/**
 * Sample plugin.
 */
Draw.loadPlugin(function(ui) {

	/**
	 * Basic adds for all backends.
	 */
	var tweet = '<a title="Enjoying draw.io? Tell the world!" ' +
                    'target="_blank" href="https://twitter.com/intent/tweet?text=' +
                    encodeURIComponent('I\'ve discovered draw.io for free online flowcharts, UML, ERD and more...') +
                    '" onclick="javascript:window.open(this.href, \'\', \'menubar=no,toolbar=no,resizable=yes,scrollbars=yes,' +
                    'left=\'+((screen.width-640)/2)+\',top=\'+((screen.height-280)/3)+\',height=280,width=640\');return false;"\'>' +
                    '<img border="0" align="absmiddle" width="18" height="18" style="margin-top:-2px;padding-right:8px;" src="' +
                    Editor.tweetImage + '"/>Enjoying draw.io? Tell the world!</a>';
        var like = '<a title="Like us on Facebook!" ' +
                    'target="_blank" href="https://www.facebook.com/drawioapp/">' +
                    '<img border="0" align="absmiddle" width="18" height="18" style="margin-top:-2px;padding-right:8px;" src="' +
                    Editor.facebookImage + '"/>Like us on Facebook!</a>';
  	var star = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAABx0RVh0U29mdHdhcmUAQWRvYmUgRmlyZXdvcmtzIENTM5jWRgMAAAQRdEVYdFhNTDpjb20uYWRvYmUueG1wADw/eHBhY2tldCBiZWdpbj0iICAgIiBpZD0iVzVNME1wQ2VoaUh6cmVTek5UY3prYzlkIj8+Cjx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IkFkb2JlIFhNUCBDb3JlIDQuMS1jMDM0IDQ2LjI3Mjk3NiwgU2F0IEphbiAyNyAyMDA3IDIyOjExOjQxICAgICAgICAiPgogICA8cmRmOlJERiB4bWxuczpyZGY9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkvMDIvMjItcmRmLXN5bnRheC1ucyMiPgogICAgICA8cmRmOkRlc2NyaXB0aW9uIHJkZjphYm91dD0iIgogICAgICAgICAgICB4bWxuczp4YXA9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC8iPgogICAgICAgICA8eGFwOkNyZWF0b3JUb29sPkFkb2JlIEZpcmV3b3JrcyBDUzM8L3hhcDpDcmVhdG9yVG9vbD4KICAgICAgICAgPHhhcDpDcmVhdGVEYXRlPjIwMDgtMDItMTdUMDI6MzY6NDVaPC94YXA6Q3JlYXRlRGF0ZT4KICAgICAgICAgPHhhcDpNb2RpZnlEYXRlPjIwMDktMDMtMTdUMTQ6MTI6MDJaPC94YXA6TW9kaWZ5RGF0ZT4KICAgICAgPC9yZGY6RGVzY3JpcHRpb24+CiAgICAgIDxyZGY6RGVzY3JpcHRpb24gcmRmOmFib3V0PSIiCiAgICAgICAgICAgIHhtbG5zOmRjPSJodHRwOi8vcHVybC5vcmcvZGMvZWxlbWVudHMvMS4xLyI+CiAgICAgICAgIDxkYzpmb3JtYXQ+aW1hZ2UvcG5nPC9kYzpmb3JtYXQ+CiAgICAgIDwvcmRmOkRlc2NyaXB0aW9uPgogICA8L3JkZjpSREY+CjwveDp4bXBtZXRhPgogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIIImu8AAAAAVdEVYdENyZWF0aW9uIFRpbWUAMi8xNy8wOCCcqlgAAAHuSURBVDiNlZJBi1JRGIbfk+fc0ZuMXorJe4XujWoMdREaA23HICj6AQeLINr0C4I27ab27VqOI9+q/sH8gMDceG1RkIwgClEXFMbRc5zTZgZURmG+5fu9PN/7Hg6wZohoh4h21nn4uqXW+q0xZgzg+SrPlTXX73uet+26bp6ICpcGaK1fua57M5vN3tZav7gUgIiSqVTqcRAEm0EQbCaTyQoRXb3Iy4hoG8CT6XSaY4xtMMaSQohMPp8v+r7vAEC3243CMGwqpfoApsaYE8uyfgM45ABOjDEvXdfNlMvlzFINAIDneY7neZVzvdlsDgaDQYtzfsjOIjtKqU+e5+0Wi0V3VV8ACMOw3+/3v3HOX0sp/7K53te11h/S6fRuoVAIhBAL76OUOm2320dRFH0VQuxJKf8BAFu+UKvVvpRKpWe2bYt5fTweq0ajQUKIN1LK43N94SMR0Y1YLLYlhBBKqQUw51wkEol7WmuzoC8FuJtIJLaUUoii6Ljb7f4yxpz6vp9zHMe2bfvacDi8BeDHKkBuNps5rVbr52QyaVuW9ZExttHpdN73ej0/Ho+nADxYCdBaV0aj0RGAz5ZlHUgpx2erR/V6/d1wOHwK4CGA/QsBnPN9AN+llH+WkqFare4R0QGAO/M6M8Ysey81/wGqa8MlVvHPNAAAAABJRU5ErkJggg==';
	var lumapps = '<a title="Collaborate on diagrams in Samepage" target="_blank" ' +
			((mxClient.IS_SF) ? 'style="margin-top:-22px;" ' : '') +
			'href="https://www.samepage.io/draw-diagram-online?SPcid=SIOF%2BDraw%2Breferral%2BDraw%2Bv1%2BNA"\>' +
			'<img border="0" align="absmiddle" width="24" height="24" style="margin-top:-2px;padding-right:8px;" ' +
			'src="' + IMAGE_PATH + '/samepage-icon-color.svg"/>Diagrams in Samepage</a>';

	var basicAds = [
		// Twitter
		'<a title="' + mxResources.get('loveIt', ['draw.io']) +
		'" target="_blank" href="https://twitter.com/intent/tweet?text=' +
		encodeURIComponent(mxResources.get('loveIt', ['www.draw.io'])) +
		'" onclick="javascript:window.open(this.href, \'\', \'menubar=no,toolbar=no,resizable=yes,scrollbars=yes,' +
		'left=\'+((screen.width-640)/2)+\',top=\'+((screen.height-280)/3)+\',height=280,width=640\');return false;"\'>' +
		'<img border="0" align="absmiddle" width="18" height="18" style="margin-top:-2px;padding-right:8px;" src="' +
		Editor.tweetImage + '"/>' + mxResources.get('loveIt', ['draw.io']) + '</a>',
		// Facebook
		'<a title="Share on Facebook" target="_blank" href="https://www.facebook.com/sharer.php?u=' +
		encodeURIComponent('https://www.draw.io') +
		'" onclick="javascript:window.open(this.href, \'\', \'menubar=no,toolbar=no,resizable=yes,scrollbars=yes,' +
		'left=\'+((screen.width-640)/2)+\',top=\'+((screen.height-520)/3)+\',height=520,width=640\');return false;"\'>' +
		'<img border="0" align="absmiddle" width="18" height="18" style="margin-top:-2px;padding-right:8px;" src="' +
		Editor.facebookImage + '"/>Share on Facebook</a>',
		// Offline App
		'<a title="draw.io Offline" href="https://www.draw.io/app" target="_blank">' +
		'<img border="0" align="absmiddle" style="margin-top:-1px;padding-right:8px;" src="images/download.png"/>' +
		'draw.io Offline</a>'
	];

	var rotate = mxUtils.bind(this, function(elt, html, delay, fn)
	{
		delay = (delay != null) ? delay : 500;
		mxUtils.setPrefixedStyle(elt.style, 'transition', 'all ' + (delay / 1000) + 's ease');
		mxUtils.setPrefixedStyle(elt.style, 'transform', 'scale(0)');
		elt.style.visibility = 'visible';
		elt.style.opacity = '0';
		
		window.setTimeout(function()
		{
			elt.innerHTML = html;
			mxUtils.setPrefixedStyle(elt.style, 'transform', 'scale(1)');
			elt.style.opacity = '1';
			
			if (fn != null)
			{
				fn();
			}
		}, delay);
	});
	
	// Stats for right footer
	var td2 = document.getElementById('geFooterItem1');
	
	if (td2 != null)
	{
		window.setTimeout(function()
		{
			rotate(td2, lumapps);
		}, 5000);
		/*td2.innerHTML = '<a id="geFooterLink1" title="#1 Rated Confluence Add-on" target="_blank" ' +
			'href="https://about.draw.io/integrations/confluence-integration/">' +
			'<img border="0" width="24" height="24" align="absmiddle" style="padding-right:10px;"' +
			'src="images/logo-confluence.png"/>#1 Rated Confluence App</a>';*/
		// TODO Fix stats for rotation
		//if (false)
		//{
			/* Replaces footer with download link
			var link = 'https://download.draw.io/';
			var lastHtml = td2.innerHTML;
			var os = 'draw.io';
	
			if (['Macintosh', 'MacIntel', 'MacPPC', 'Mac68K'].indexOf(navigator.platform) >= 0)
			{
				os += ' for macOS';
			}
			else if (['Win32', 'Win64', 'Windows', 'WinCE'].indexOf(navigator.platform) >= 0)
			{
				os += ' for Windows';
			}
			else if (/\bCrOS\b/.test(navigator.userAgent))
			{
				os += ' for Chrome OS';
			}
			else if (/Linux/.test(navigator.platform))
			{
				os += ' for Linux';
			}
			else
			{
				os += ' Desktop';
			}
	
			td2.innerHTML = '<a title="' + os + '" href="' + link + '" target="_blank">' +
				'<img border="0" align="absmiddle" style="margin-top:-4px;" width="24" height="24" ' +
				'src="images/drawlogo48.png"/>&nbsp;&nbsp;' + os + '</a>';*/
			
//			window.setInterval(mxUtils.bind(this, function()
//			{
//				var temp = lastHtml;
//				lastHtml = td2.innerHTML;
//				rotate(td2, temp);
//			}), 300000);
		//}
	}

	// Changes left footer
	var td = document.getElementById('geFooterItem2');
	
	if (td != null && mxClient.IS_SVG)
	{
		var last = [td.innerHTML];
		
		/*if (mxSettings.getOpenCounter() > 10 && urlParams['embed'] != '1')
		{
			last.push('<a title="' + mxResources.get('loveIt', ['draw.io']) +
				'" target="_blank" href="https://twitter.com/intent/tweet?text=' +
				encodeURIComponent(mxResources.get('loveIt', ['www.draw.io'])) +
				'" onclick="javascript:window.open(this.href, \'\', \'menubar=no,toolbar=no,resizable=yes,scrollbars=yes,' +
				'left=\'+((screen.width-640)/2)+\',top=\'+((screen.height-280)/3)+\',height=280,width=640\');return false;"\'>' +
				'<img border="0" align="absmiddle" width="18" height="18" style="margin-top:-2px;padding-right:8px;" src="' +
				Editor.tweetImage + '"/>' + mxResources.get('loveIt', ['draw.io']) + '</a>')
		}*/
		
		/****************/
		/** Quickstart **/
		/****************/
		// rotate(td, '<a title="Desktop" target="_blank"' +
	//		'href="https://www.facebook.com/drawioapp/posts/1748776785146850"> ' +
	//	        '<img border="0" align="absmiddle" style="margin-top:-4px;padding-right:8px;" width="24" height="24" ' +
	//	        'src="images/drawlogo48.png"/>Use draw.io offline - draw.io Desktop</a>')
		
		// Restores last message after click
		mxEvent.addListener(td, 'click', function()
		{
			if (last.length > 0)
			{
				rotate(td, last.pop());
			}
		});
		
		// Waits for login with Google account and shows review link
		var userChanged = function()
		{
			var user = ui.drive.getUser();
			
			if (user != null && user.email != null) // && user.email.substring(user.email.length - 10) == '@gmail.com')
			{
				last.push(td.innerHTML);
				
				window.setTimeout(function()
				{
					rotate(td, '<a title="Drive Connector" href="https://about.draw.io/google-drive-connector-for-confluence-cloud/" target="_blank">' +
						'Embed diagrams in Confluence</a>');
				}, 1000);
			}
		};

		var install = function()
		{
			if (ui.drive != null)
			{
				if (ui.drive.getUser() != null)
				{
					userChanged();
				}
				else
				{
					ui.drive.addListener('userChanged', userChanged);
				}
			}
		};
		
		if (ui.drive != null)
		{
			install();
		}
		else
		{
			ui.addListener('clientLoaded', install);
		}
	}

// TODO: Fix stats do not work with rotation
//		basicAds.push(td.innerHTML);
//		var adsHtml = basicAds;
//		var lastAd = adsHtml.length - 1;
//		var thread2 = null;
//		var thread = null;
//
//		function updateAd(index, delay)
//		{
//			if (thread != null)
//			{
//				window.clearTimeout(thread);
//				thread = null;
//			}
//	
//			if (thread2 != null)
//			{
//				window.clearTimeout(thread2);
//				thread2 = null;
//			}
//			
//			if (adsHtml.length == 0)
//			{
//				td.style.display = 'none';
//				td.innerHTML = '';
//			}
//			else
//			{
//				var schedule = mxUtils.bind(this, function()
//				{
//					if (adsHtml.length > 0)
//					{
//						thread = window.setTimeout(mxUtils.bind(this, function()
//						{
//							var tmp = 0;
//							
//							if (adsHtml.length > 1)
//							{
//								tmp = Math.round(Math.random() * (adsHtml.length - 2));
//								
//								if (tmp == lastAd)
//								{
//									tmp++;
//								}
//							}
//							
//							updateAd(tmp);
//						}), 180000);
//					}
//				});
//			
//				if (index != lastAd)
//				{
//					rotate(td, adsHtml[index], delay, function()
//					{
//						lastAd = index;
//						schedule();
//					});
//				}
//				else
//				{
//					schedule();
//				}
//			}
//		};
//		
//		mxEvent.addListener(td, 'click', mxUtils.bind(this, function()
//		{
//			adsHtml.splice(lastAd, 1);
//			lastAd = null;
//			updateAd(0);
//		}));
//	
//		if (mxSettings.getOpenCounter() > 10 && urlParams['embed'] != '1')
//		{
//			thread2 = window.setTimeout(mxUtils.bind(this, function()
//			{
//				updateAd(0);
//			}), 15000);
//		}
//		else
//		{
//			updateAd(adsHtml.length - 1);
//		}
});
