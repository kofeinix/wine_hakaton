export function useWineFormat() {
  function mainPhoto(wine) {
    return wine?.image_url || wine?.photos?.find((photo) => photo.is_main)?.url || wine?.photos?.[0]?.url || "";
  }

  function realPhotos(wine) {
    return wine?.photos || [];
  }

  function generatedPhotos(wine) {
    return wine?.generated_photos || [];
  }

  function formatPercent(value) {
    if (typeof value !== "number") return "—";
    return `${Math.round(value * 100)}%`;
  }

  function formatPrice(wine) {
    if (!wine?.price) return "—";
    return `${Number(wine.price).toLocaleString("ru-RU")} ${wine.currency || "RUB"}`;
  }

  function regionLine(wine) {
    return [wine?.country, wine?.region].filter(Boolean).join(", ") || "—";
  }

  function grapeLine(wine) {
    return wine?.grapes?.length ? wine.grapes.join(", ") : "—";
  }

  function categoryLine(wine) {
    return [wine?.color, wine?.sugar, wine?.year].filter(Boolean).join(" · ") || "—";
  }

  function alcoholLine(wine) {
    if (!wine?.alcohol) return "—";
    if (typeof wine.alcohol === "number") return `${wine.alcohol}%`;
    return wine.alcohol;
  }

  function servingTemperature(wine) {
    return wine?.serving_temperature || "—";
  }

  function foodPairing(wine) {
    return wine?.food_pairings || [];
  }

  function shadeLine(wine) {
    return wine?.shade || "—";
  }

  function prettyBytes(bytes) {
    if (!bytes) return "";
    if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} КБ`;
    return `${(bytes / 1024 / 1024).toFixed(1)} МБ`;
  }

  function formatDate(value) {
    return value
      ? new Date(value).toLocaleString("ru-RU", {
          day: "2-digit",
          month: "short",
          hour: "2-digit",
          minute: "2-digit",
        })
      : "";
  }

  return {
    alcoholLine,
    categoryLine,
    foodPairing,
    formatDate,
    formatPercent,
    formatPrice,
    generatedPhotos,
    grapeLine,
    mainPhoto,
    prettyBytes,
    realPhotos,
    regionLine,
    servingTemperature,
    shadeLine,
  };
}
