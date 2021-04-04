/**
 * Template Name: Vesperr - v2.3.0
 * Template URL: https://bootstrapmade.com/vesperr-free-bootstrap-template/
 * Author: BootstrapMade.com
 * License: https://bootstrapmade.com/license/
 */

async function postData(url = "", data = {}) {
  // Default options are marked with *
  const response = await fetch(url, {
    method: "POST", // *GET, POST, PUT, DELETE, etc.
    mode: "cors", // no-cors, *cors, same-origin
    cache: "no-cache", // *default, no-cache, reload, force-cache, only-if-cached
    credentials: "same-origin", // include, *same-origin, omit
    headers: {
      "Content-Type": "application/json",
      // 'Content-Type': 'application/x-www-form-urlencoded',
    },
    redirect: "follow", // manual, *follow, error
    referrerPolicy: "no-referrer", // no-referrer, *no-referrer-when-downgrade, origin, origin-when-cross-origin, same-origin, strict-origin, strict-origin-when-cross-origin, unsafe-url
    body: JSON.stringify(data), // body data type must match "Content-Type" header
  });
  return response.json(); // parses JSON response into native JavaScript objects
}

$(function () {
  var INDEX = 0;
  $("#chat-submit").click(function (e) {
    e.preventDefault();
    var msg = $("#chat-input").val();
    if (msg.trim() == "") {
      return false;
    }
    generate_message(msg, "self");
    var buttons = [
      {
        name: "Existing User",
        value: "existing",
      },
      {
        name: "New User",
        value: "new",
      },
    ];
    setTimeout(function () {
      var res;
      postData("https://rasatestchatbot.herokuapp.com/webhooks/rest/webhook", {
        message: msg,
        sender: 1,
      }).then((data) => {
        generate_message(data[0].text, "user"); // JSON data parsed by `data.json()` call
      });
    }, 1000);
  });

  function generate_message(msg, type) {
    INDEX++;
    var str = "";
    str += "<div id='cm-msg-" + INDEX + "' class=\"chat-msg " + type + '">';
    str += '          <span class="msg-avatar">';
    str +=
      '            <img src="https://image.crisp.im/avatar/operator/196af8cc-f6ad-4ef7-afd1-c45d5231387c/240/?1483361727745">';
    str += "          </span>";
    str += '          <div class="cm-msg-text">';
    str += msg;
    str += "          </div>";
    str += "        </div>";
    $(".chat-logs").append(str);
    $("#cm-msg-" + INDEX)
      .hide()
      .fadeIn(300);
    if (type == "self") {
      $("#chat-input").val("");
    }
    $(".chat-logs")
      .stop()
      .animate(
        {
          scrollTop: $(".chat-logs")[0].scrollHeight,
        },
        1000
      );
  }

  function generate_button_message(msg, buttons) {
    /* Buttons should be object array
      [
        {
          name: 'Existing User',
          value: 'existing'
        },
        {
          name: 'New User',
          value: 'new'
        }
      ]
    */
    INDEX++;
    var btn_obj = buttons
      .map(function (button) {
        return (
          '              <li class="button"><a href="javascript:;" class="btn btn-primary chat-btn" chat-value="' +
          button.value +
          '">' +
          button.name +
          "</a></li>"
        );
      })
      .join("");
    var str = "";
    str += "<div id='cm-msg-" + INDEX + '\' class="chat-msg user">';
    str += '          <span class="msg-avatar">';
    str +=
      '            <img src="https://image.crisp.im/avatar/operator/196af8cc-f6ad-4ef7-afd1-c45d5231387c/240/?1483361727745">';
    str += "          </span>";
    str += '          <div class="cm-msg-text">';
    str += msg;
    str += "          </div>";
    str += '          <div class="cm-msg-button">';
    str += "            <ul>";
    str += btn_obj;
    str += "            </ul>";
    str += "          </div>";
    str += "        </div>";
    $(".chat-logs").append(str);
    $("#cm-msg-" + INDEX)
      .hide()
      .fadeIn(300);
    $(".chat-logs")
      .stop()
      .animate(
        {
          scrollTop: $(".chat-logs")[0].scrollHeight,
        },
        1000
      );
    $("#chat-input").attr("disabled", true);
  }

  $(document).delegate(".chat-btn", "click", function () {
    var value = $(this).attr("chat-value");
    var name = $(this).html();
    $("#chat-input").attr("disabled", false);
    generate_message(name, "self");
  });

  $("#chat-circle").click(function () {
    $("#chat-circle").toggle("scale");
    $(".chat-box").toggle("scale");
  });

  $(".chat-box-toggle").click(function () {
    $("#chat-circle").toggle("scale");
    $(".chat-box").toggle("scale");
  });
});
!(function ($) {
  "use strict";

  // Smooth scroll for the navigation menu and links with .scrollto classes
  var scrolltoOffset = $("#header").outerHeight() - 15;
  $(document).on(
    "click",
    ".nav-menu a, .mobile-nav a, .scrollto",
    function (e) {
      if (
        location.pathname.replace(/^\//, "") ==
          this.pathname.replace(/^\//, "") &&
        location.hostname == this.hostname
      ) {
        var target = $(this.hash);
        if (target.length) {
          e.preventDefault();

          var scrollto = target.offset().top - scrolltoOffset;

          if ($(this).attr("href") == "#header") {
            scrollto = 0;
          }

          $("html, body").animate(
            {
              scrollTop: scrollto,
            },
            1500,
            "easeInOutExpo"
          );

          if ($(this).parents(".nav-menu, .mobile-nav").length) {
            $(".nav-menu .active, .mobile-nav .active").removeClass("active");
            $(this).closest("li").addClass("active");
          }

          if ($("body").hasClass("mobile-nav-active")) {
            $("body").removeClass("mobile-nav-active");
            $(".mobile-nav-toggle i").toggleClass(
              "icofont-navigation-menu icofont-close"
            );
            $(".mobile-nav-overly").fadeOut();
          }
          return false;
        }
      }
    }
  );

  // Activate smooth scroll on page load with hash links in the url
  $(document).ready(function () {
    if (window.location.hash) {
      var initial_nav = window.location.hash;
      if ($(initial_nav).length) {
        var scrollto = $(initial_nav).offset().top - scrolltoOffset;
        $("html, body").animate(
          {
            scrollTop: scrollto,
          },
          1500,
          "easeInOutExpo"
        );
      }
    }
  });

  // Mobile Navigation
  if ($(".nav-menu").length) {
    var $mobile_nav = $(".nav-menu").clone().prop({
      class: "mobile-nav d-lg-none",
    });
    $("body").append($mobile_nav);
    $("body").prepend(
      '<button type="button" class="mobile-nav-toggle d-lg-none"><i class="icofont-navigation-menu"></i></button>'
    );
    $("body").append('<div class="mobile-nav-overly"></div>');

    $(document).on("click", ".mobile-nav-toggle", function (e) {
      $("body").toggleClass("mobile-nav-active");
      $(".mobile-nav-toggle i").toggleClass(
        "icofont-navigation-menu icofont-close"
      );
      $(".mobile-nav-overly").toggle();
    });

    $(document).on("click", ".mobile-nav .drop-down > a", function (e) {
      e.preventDefault();
      $(this).next().slideToggle(300);
      $(this).parent().toggleClass("active");
    });

    $(document).click(function (e) {
      var container = $(".mobile-nav, .mobile-nav-toggle");
      if (!container.is(e.target) && container.has(e.target).length === 0) {
        if ($("body").hasClass("mobile-nav-active")) {
          $("body").removeClass("mobile-nav-active");
          $(".mobile-nav-toggle i").toggleClass(
            "icofont-navigation-menu icofont-close"
          );
          $(".mobile-nav-overly").fadeOut();
        }
      }
    });
  } else if ($(".mobile-nav, .mobile-nav-toggle").length) {
    $(".mobile-nav, .mobile-nav-toggle").hide();
  }

  // Navigation active state on scroll
  var nav_sections = $("section");
  var main_nav = $(".nav-menu, #mobile-nav");

  $(window).on("scroll", function () {
    var cur_pos = $(this).scrollTop() + 200;

    nav_sections.each(function () {
      var top = $(this).offset().top,
        bottom = top + $(this).outerHeight();

      if (cur_pos >= top && cur_pos <= bottom) {
        if (cur_pos <= bottom) {
          main_nav.find("li").removeClass("active");
        }
        main_nav
          .find('a[href="#' + $(this).attr("id") + '"]')
          .parent("li")
          .addClass("active");
      }
      if (cur_pos < 300) {
        $(".nav-menu ul:first li:first").addClass("active");
      }
    });
  });

  // Toggle .header-scrolled class to #header when page is scrolled
  $(window).scroll(function () {
    if ($(this).scrollTop() > 100) {
      $("#header").addClass("header-scrolled");
    } else {
      $("#header").removeClass("header-scrolled");
    }
  });

  if ($(window).scrollTop() > 100) {
    $("#header").addClass("header-scrolled");
  }

  // Back to top button
  $(window).scroll(function () {
    if ($(this).scrollTop() > 100) {
      $(".back-to-top").fadeIn("slow");
    } else {
      $(".back-to-top").fadeOut("slow");
    }
  });

  $(".back-to-top").click(function () {
    $("html, body").animate(
      {
        scrollTop: 0,
      },
      1500,
      "easeInOutExpo"
    );
    return false;
  });

  // jQuery counterUp
  $('[data-toggle="counter-up"]').counterUp({
    delay: 10,
    time: 1000,
  });

  // Testimonials carousel (uses the Owl Carousel library)
  $(".testimonials-carousel").owlCarousel({
    autoplay: true,
    dots: true,
    loop: true,
    responsive: {
      0: {
        items: 1,
      },
      768: {
        items: 1,
      },
      900: {
        items: 2,
      },
    },
  });

  // Porfolio isotope and filter
  $(window).on("load", function () {
    var portfolioIsotope = $(".portfolio-container").isotope({
      itemSelector: ".portfolio-item",
      layoutMode: "fitRows",
    });

    $("#portfolio-flters li").on("click", function () {
      $("#portfolio-flters li").removeClass("filter-active");
      $(this).addClass("filter-active");

      portfolioIsotope.isotope({
        filter: $(this).data("filter"),
      });
      aos_init();
    });

    // Initiate venobox (lightbox feature used in portofilo)
    $(document).ready(function () {
      $(".venobox").venobox();
    });
  });

  // Portfolio details carousel
  $(".portfolio-details-carousel").owlCarousel({
    autoplay: true,
    dots: true,
    loop: true,
    items: 1,
  });

  // Init AOS
  function aos_init() {
    AOS.init({
      duration: 1000,
      easing: "ease-in-out",
      once: true,
      mirror: false,
    });
  }
  $(window).on("load", function () {
    aos_init();
  });
})(jQuery);

// Button for nearby place
const div_ele = document.getElementById("btn_fetch");

div_ele.style.cursor = "pointer";
div_ele.onclick = function () {
  let a = document.getElementById("query_search").value;

  fetch(
    "https://api.tomtom.com/search/2/search/" +
      a +
      " hospital.json?key=X1cR6CbyjSYtqH5fSF6IPkEycq88z5Xh&limit=100&radius=5000&lat=23.0979733&lon=72.5469074"
  )
    .then((res) => res.json())
    .then((out) => {
      var a = out.results;
      const ul = document.getElementById("nearby_places");
      for (let x of a) {
        const li = document.createElement("li");
        li.innerHTML = x.poi.name;
        ul.appendChild(li);
      }
      //console.log();
    })
    .catch((err) => console.error(err));
};

var div_submit = document.getElementById("submit_form");
div_submit.onclick = function () {
  let s1 = $("#symp1").text();
  let s2 = $("#symp2").text();
  let s3 = $("#symp3").text();
  let s4 = $("#symp4").text();
  let s5 = $("#symp5").text();
  if (
    s1 !== "First option" &&
    s2 !== "First option" &&
    s3 !== "First option" &&
    s4 !== "First option" &&
    s5 !== "First option"
  ) {
    document.getElementById("ssymp1").setAttribute("value", s1);
    document.getElementById("ssymp2").setAttribute("value", s2);
    document.getElementById("ssymp3").setAttribute("value", s3);
    document.getElementById("ssymp4").setAttribute("value", s4);
    document.getElementById("ssymp5").setAttribute("value", s5);
  }

  $("#disease_form").submit();
};

$("#clear_all_ele").click(function () {
  $("#nearby_places").empty();
});
$(document).ready(function () {
  const select = document.querySelectorAll(".selectBtn");
  const option = document.querySelectorAll(".option");
  let index = 1;

  select.forEach((a) => {
    a.addEventListener("click", (b) => {
      const next = b.target.nextElementSibling;
      next.classList.toggle("toggle");
      next.style.zIndex = index++;
    });
  });
  option.forEach((a) => {
    a.addEventListener("click", (b) => {
      b.target.parentElement.classList.remove("toggle");

      const parent = b.target.closest(".select").children[0];
      parent.setAttribute("data-type", b.target.getAttribute("data-type"));
      parent.innerText = b.target.innerText;
    });
  });
  $("#pp").click(function () {
    console.log($("#first").text());
  });
});

$("#disease_form").submit(function (e) {
  // Get all the forms elements and their values in one step
  var values = $(this).serialize();
  console.log(values);
  // e.preventDefault();
});

