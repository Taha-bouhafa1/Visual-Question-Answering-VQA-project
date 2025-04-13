document.addEventListener("DOMContentLoaded", function () {
    "use strict";

    /* Preloader */
    var loaderBg = document.querySelector('.loader_bg');
    if (loaderBg) {
        loaderBg.style.display = 'block'; // Show the loader when the DOM content starts loading
    }

    /* Hide loader after delay */
    setTimeout(function () {
        if (loaderBg) {
            loaderBg.style.display = 'none'; // Hide the loader after a delay
        }
    }, 1500);

    /* Tooltip */
    var tooltips = document.querySelectorAll('[data-toggle="tooltip"]');
    tooltips.forEach(function (tooltip) {
        tooltip.addEventListener('mouseover', function () {
            // Your tooltip logic here
        });
    });

    /* Toggle sidebar */
    var sidebarCollapse = document.getElementById('sidebarCollapse');
    if (sidebarCollapse) {
        sidebarCollapse.addEventListener('click', function () {
            var sidebar = document.getElementById('sidebar');
            if (sidebar) {
                sidebar.classList.toggle('active');
            }
            this.classList.toggle('active');
        });
    }
});
