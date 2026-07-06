When you talk about a "rich data format" for a local business, you are looking for **LocalBusiness Schema Markup** (structured data). This is a specific format of code (usually written in JSON-LD) that you add to a website so search engines like Google can easily understand who you are, what you do, and where you are located.

Implementing this correctly can land your business in Google's "Local Pack" (the map results) and give you rich snippets in search results.

Here is the ultimate, fully-optimized JSON-LD template for a local business.

### The Ultimate Local Business JSON-LD Template

You can copy this, replace the placeholder data with your actual business info, and paste it into the `<head>` section of your website.

```json
{
  "@context": "https://schema.org",
  "@type": "LocalBusiness",
  "@id": "https://www.example.com/#organization",
  "name": "The Daily Grind Cafe",
  "image": [
    "https://www.example.com/images/1x1/cafe-front.jpg",
    "https://www.example.com/images/4x3/cafe-interior.jpg",
    "https://www.example.com/images/16x9/coffee-pouring.jpg"
  ],
  "telePhone": "+1-555-123-4567",
  "url": "https://www.example.com",
  "logo": "https://www.example.com/images/logo.png",
  "sameAs": [
    "https://www.facebook.com/thedailygrindcafe",
    "https://www.instagram.com/thedailygrindcafe",
    "https://www.linkedin.com/company/thedailygrindcafe"
  ],
  "address": {
    "@type": "PostalAddress",
    "streetAddress": "123 Main Street, Suite A",
    "addressLocality": "Austin",
    "addressRegion": "TX",
    "postalCode": "78701",
    "addressCountry": "US"
  },
  "geo": {
    "@type": "GeoCoordinates",
    "latitude": 30.2672,
    "longitude": -97.7431
  },
  "openingHoursSpecification": [
    {
      "@type": "OpeningHoursSpecification",
      "dayOfWeek": [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday"
      ],
      "opens": "07:00",
      "closes": "18:00"
    },
    {
      "@type": "OpeningHoursSpecification",
      "dayOfWeek": [
        "Saturday",
        "Sunday"
      ],
      "opens": "08:00",
      "closes": "16:00"
    }
  ],
  "priceRange": "$$",
  "paymentAccepted": "Cash, Credit Card, Apple Pay",
  "currenciesAccepted": "USD",
  "areaServed": {
    "@type": "GeoCircle",
    "geoMidpoint": {
      "@type": "GeoCoordinates",
      "latitude": 30.2672,
      "longitude": -97.7431
    },
    "geoRadius": "5000"
  }
}

```

---

### Core Fields Explained

To get the most SEO value out of this rich data, make sure you understand these critical fields:

* **`@type`**: While `LocalBusiness` works, it is best to be as specific as possible if Schema.org supports it (e.g., `Restaurant`, `AutomobileRepair`, `Dentist`, `HairSalon`, `RealEstateAgent`).
* **`@id`**: A unique URL identifier for your business. It prevents Google from getting confused if you have multiple branches or overlapping data.
* **`geo`**: The exact latitude and longitude of your physical front door. This is incredibly powerful for local map rankings.
* **`sameAs`**: This tells Google, *"Hey, this website, this Facebook page, and this Instagram account all belong to the exact same entity."* It builds massive trust.
* **`areaServed`**: Defines your service radius (in meters) if you are a delivery business or service-area business (like a plumber or electrician).

---

### How to Test and Deploy Your Rich Data

1. **Customize:** Update the template above with your NAP (Name, Address, Phone number) exactly as it appears on your Google Business Profile. consistency is key!
2. **Test:** Before publishing, copy your code and paste it into [Google's Rich Results Test](https://search.google.com/test/rich-results) to ensure there are no syntax errors or missing required fields.
3. **Deploy:** Paste the final code into the `<head>` of your homepage. If you use WordPress, you can use plugins like *RankMath* or *Yoast*, or just inject it using a header/footer plugin.