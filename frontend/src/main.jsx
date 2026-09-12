import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API_SEARCH_URL = "/api/v1/search/image/extended";
const API_WINES_URL = "/api/v1/wines";

function formatPercent(value) {
  return typeof value === "number" ? `${Math.round(value * 100)}%` : "0%";
}

function formatPrice(wine) {
  if (!wine?.price) {
    return null;
  }
  return `${wine.price.toLocaleString("ru-RU")} ${wine.currency || ""}`.trim();
}

function wineRating(wine) {
  return wine?.rating || wine?.average_rating || 0;
}

function StarRating({ value }) {
  const rating = Number(value) || 0;
  const rounded = Math.round(rating);

  if (!rating) {
    return null;
  }

  return (
    <span className="stars" aria-label={`Рейтинг ${rating} из 5`}>
      <span className="star-icons" aria-hidden="true">
        {Array.from({ length: 5 }, (_, index) => (
          <span key={index} className={index < rounded ? "filled" : ""}>
            ★
          </span>
        ))}
      </span>
      <span className="rating-value">{rating.toFixed(2).replace(/\.?0+$/, "")}</span>
    </span>
  );
}

function mainPhoto(wine) {
  return wine?.image_url || wine?.photos?.find((photo) => photo.is_main)?.url || wine?.photos?.[0]?.url || null;
}

function getWineFacts(wine) {
  const rating = wineRating(wine);
  return [
    ["Производитель", wine.producer],
    ["Страна", wine.country],
    ["Регион", wine.region],
    ["Год", wine.vintage],
    ["Тип", wine.wine_type || wine.style],
    ["Цвет", wine.color],
    ["Сахар", wine.style && wine.color ? wine.style.replace(wine.color, "").trim() : null],
    ["Сорт винограда", wine.grapes?.join(", ")],
    ["Алкоголь", wine.alcohol],
    ["Рейтинг", rating ? <StarRating value={rating} /> : null],
    ["Оценок", wine.ratings_count || null],
    ["Цена", formatPrice(wine)],
    ["Подача", wine.serving_temperature],
    ["Оттенок", wine.shade],
  ].filter(([, value]) => value);
}

function WineFacts({ wine }) {
  const facts = getWineFacts(wine);

  return (
    <dl className="wine-facts">
      {facts.map(([label, value]) => (
        <React.Fragment key={label}>
          <dt>{label}</dt>
          <dd>{value}</dd>
        </React.Fragment>
      ))}
    </dl>
  );
}

function PhotoGallery({ wine }) {
  const photos = wine?.photos || [];
  const [activePhoto, setActivePhoto] = useState(mainPhoto(wine));

  useEffect(() => {
    setActivePhoto(mainPhoto(wine));
  }, [wine]);

  if (!photos.length && !activePhoto) {
    return <div className="hero-photo-placeholder">Фото вина пока нет</div>;
  }

  return (
    <div className="photo-gallery">
      <div className="hero-photo">
        <img src={activePhoto} alt={wine.name} />
      </div>
      {photos.length > 1 && (
        <div className="thumbs" aria-label="Фотографии вина">
          {photos.map((photo) => (
            <button
              key={photo.object_name || photo.url}
              type="button"
              className={photo.url === activePhoto ? "active" : ""}
              onClick={() => setActivePhoto(photo.url)}
            >
              <img src={photo.url} alt={photo.filename || wine.name} />
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function WineDetails({ match, mode = "best" }) {
  const wine = match.wine;
  const wineType = wine.wine_type || null;
  const quickFacts = getWineFacts(wine).filter(([label]) => !["Рейтинг", "Оценок"].includes(label));

  return (
    <article className="wine-details">
      <PhotoGallery wine={wine} />
      <div className="details-content">
        <div className="details-heading">
          <div className="match-kicker">{mode === "best" ? "Лучшее совпадение" : "Выбранный вариант"} · {formatPercent(match.score)}</div>
          <StarRating value={wineRating(wine)} />
        </div>
        <h2>{wine.name}</h2>
        {wine.producer && <p className="lead">{wine.producer}</p>}

        {(wineType || wine.country || wine.region || wine.vintage) && (
          <div className="tags">
            {[wineType, wine.country, wine.region, wine.vintage].filter(Boolean).map((tag) => (
              <span key={tag}>{tag}</span>
            ))}
          </div>
        )}

        <div className="mobile-fact-cards" aria-label="Характеристики вина">
          {quickFacts.map(([label, value]) => (
            <div className="fact-card" key={label}>
              <span>{label}</span>
              <strong>{value}</strong>
            </div>
          ))}
        </div>

        <WineFacts wine={wine} />

        {wine.description && (
          <section className="description-block">
            <h3>Описание</h3>
            <p>{wine.description}</p>
          </section>
        )}

        {wine.food_pairings?.length > 0 && (
          <section className="description-block">
            <h3>Сочетания</h3>
            <p>{wine.food_pairings.join(", ")}</p>
          </section>
        )}

        {wine.url && (
          <a className="source-link" href={wine.url} target="_blank" rel="noreferrer">
            Открыть источник
          </a>
        )}
      </div>
    </article>
  );
}

function SimilarWineCard({ match, onSelect, isLoading }) {
  const wine = match.wine;
  const image = mainPhoto(wine);
  const meta = [wine.country, wine.region, wine.vintage].filter(Boolean).join(" · ");

  return (
    <button className="similar-card" type="button" onClick={onSelect} disabled={isLoading}>
      <div className="similar-image">
        {image ? (
          <img className="similar-image-photo" src={image} alt={wine.name} />
        ) : (
          <span>Нет фото</span>
        )}
      </div>
      <div>
        <div className="similar-score">{isLoading ? "Загрузка..." : formatPercent(match.score)}</div>
        <h3>{wine.name}</h3>
        {wine.producer && <p>{wine.producer}</p>}
        {meta && <p>{meta}</p>}
        <div className="similar-meta">
          <StarRating value={wineRating(wine)} />
          {wine.wine_type && <span>{wine.wine_type}</span>}
        </div>
      </div>
    </button>
  );
}

function ResultToolbar({ hasResults, hasSimilar, isSelected, onBack, onNewSearch }) {
  if (!hasResults) {
    return null;
  }

  return (
    <div className="result-toolbar" aria-label="Навигация по результатам">
      <button type="button" onClick={onNewSearch}>
        Новый поиск
      </button>
      {isSelected && (
        <button type="button" onClick={onBack}>
          Лучшее
        </button>
      )}
      {hasSimilar && (
        <a href="#similar-results">
          Похожие
        </a>
      )}
    </div>
  );
}

function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [limit, setLimit] = useState(5);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [response, setResponse] = useState(null);
  const [selectedMatch, setSelectedMatch] = useState(null);
  const [selectedWineIdLoading, setSelectedWineIdLoading] = useState(null);
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [cameraStream, setCameraStream] = useState(null);
  const fileInputRef = useRef(null);
  const cameraInputRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const resultsPanelRef = useRef(null);

  const results = response?.results || [];
  const bestMatch = results[0] || null;
  const similarMatches = results.slice(1);
  const uploadTitle = useMemo(() => file?.name || "Снимите этикетку или выберите фото", [file]);

  function resetSearch() {
    setFile(null);
    setResponse(null);
    setSelectedMatch(null);
    setSelectedWineIdLoading(null);
    setError("");

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      setPreviewUrl(null);
    }

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
    if (cameraInputRef.current) {
      cameraInputRef.current.value = "";
    }
  }

  function setSelectedFile(nextFile) {
    if (!nextFile) {
      return;
    }

    const hasImageType = nextFile.type.startsWith("image/");
    const hasImageName = /\.(avif|heic|heif|jpe?g|png|webp)$/i.test(nextFile.name);

    if (!hasImageType && !hasImageName) {
      setError("Выберите файл изображения.");
      return;
    }

    setFile(nextFile);
    setResponse(null);
    setSelectedMatch(null);
    setSelectedWineIdLoading(null);
    setError("");

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(URL.createObjectURL(nextFile));
  }

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  useEffect(() => {
    if (videoRef.current && cameraStream) {
      videoRef.current.srcObject = cameraStream;
    }
  }, [cameraStream]);

  useEffect(() => {
    return () => {
      cameraStream?.getTracks().forEach((track) => track.stop());
    };
  }, [cameraStream]);

  async function openCamera() {
    if (!navigator.mediaDevices?.getUserMedia || !window.isSecureContext) {
      cameraInputRef.current?.click();
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: "environment" },
          width: { ideal: 1600 },
          height: { ideal: 1200 },
        },
        audio: false,
      });

      setCameraStream(stream);
      setIsCameraOpen(true);
      setError("");
    } catch (err) {
      setError("Не удалось открыть камеру. Проверьте разрешение браузера или выберите фото из галереи.");
      cameraInputRef.current?.click();
    }
  }

  function closeCamera() {
    cameraStream?.getTracks().forEach((track) => track.stop());
    setCameraStream(null);
    setIsCameraOpen(false);
  }

  function capturePhoto() {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!video || !canvas || !video.videoWidth || !video.videoHeight) {
      setError("Камера еще не готова. Попробуйте еще раз.");
      return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(
      (blob) => {
        if (!blob) {
          setError("Не удалось сделать снимок.");
          return;
        }

        setSelectedFile(new File([blob], `wine-photo-${Date.now()}.jpg`, { type: "image/jpeg" }));
        closeCamera();
      },
      "image/jpeg",
      0.92,
    );
  }

  async function submitSearch(event) {
    event.preventDefault();

    if (!file) {
      setError("Сначала загрузите фото вина.");
      return;
    }

    const formData = new FormData();
    formData.append("image", file);

    setIsLoading(true);
    setError("");
    setResponse(null);
    setSelectedMatch(null);
    setSelectedWineIdLoading(null);

    try {
      const res = await fetch(`${API_SEARCH_URL}?limit=${limit}`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const body = await res.json().catch(() => null);
        throw new Error(body?.detail || `API вернул ${res.status}`);
      }

      setResponse(await res.json());
    } catch (err) {
      setError(err.message || "Не удалось выполнить поиск.");
    } finally {
      setIsLoading(false);
    }
  }

  async function openSimilarWine(match) {
    const wineId = match.wine.id;
    setSelectedWineIdLoading(wineId);
    setError("");

    try {
      const res = await fetch(`${API_WINES_URL}/${wineId}`);
      if (!res.ok) {
        const body = await res.json().catch(() => null);
        throw new Error(body?.detail || `API вернул ${res.status}`);
      }

      const wine = await res.json();
      setSelectedMatch({ ...match, wine });
      requestAnimationFrame(() => {
        resultsPanelRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    } catch (err) {
      setError(err.message || "Не удалось загрузить карточку вина.");
    } finally {
      setSelectedWineIdLoading(null);
    }
  }

  return (
    <main className="app-shell">
      <section className="search-panel">
        <div className="brand">
          <img className="brand-logo" src="/assets/svoe-vino-logo.svg" alt="Свое Вино" />
          <div>
            <h1>Свое Вино</h1>
            <p>Хакатон - Поиск вина по фото</p>
          </div>
        </div>

        <form onSubmit={submitSearch} className="upload-form">
          <button
            type="button"
            className={`drop-zone ${isDragging ? "dragging" : ""}`}
            onClick={() => fileInputRef.current?.click()}
            onDragEnter={(event) => {
              event.preventDefault();
              setIsDragging(true);
            }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={() => setIsDragging(false)}
            onDrop={(event) => {
              event.preventDefault();
              setIsDragging(false);
              setSelectedFile(event.dataTransfer.files?.[0]);
            }}
          >
            {previewUrl ? <img src={previewUrl} alt="Загруженное фото" /> : <span className="upload-icon">+</span>}
            <strong>{uploadTitle}</strong>
            <small>JPG, PNG или HEIC, лучше крупно и без бликов</small>
          </button>

          <input
            ref={cameraInputRef}
            className="visually-hidden"
            type="file"
            accept="image/*,.heic,.heif"
            capture="environment"
            onChange={(event) => setSelectedFile(event.target.files?.[0])}
          />
          <input
            ref={fileInputRef}
            className="visually-hidden"
            type="file"
            accept="image/*,.heic,.heif"
            onChange={(event) => setSelectedFile(event.target.files?.[0])}
          />

          <div className="file-actions">
            <button type="button" onClick={openCamera}>
              Снять фото
            </button>
            <button type="button" onClick={() => fileInputRef.current?.click()}>
              Выбрать из галереи
            </button>
          </div>

          <div className="controls">
            <label>
              Результатов
              <select value={limit} onChange={(event) => setLimit(Number(event.target.value))}>
                {[3, 5, 10, 15, 20].map((value) => (
                  <option key={value} value={value}>
                    {value}
                  </option>
                ))}
              </select>
            </label>
            <button className="primary-button" type="submit" disabled={isLoading}>
              {isLoading ? "Ищем..." : "Найти вино"}
            </button>
          </div>
        </form>

        {error && <div className="error">{error}</div>}
      </section>

      {isCameraOpen && (
        <div className="camera-modal" role="dialog" aria-modal="true" aria-label="Съемка фото">
          <div className="camera-sheet">
            <video ref={videoRef} autoPlay playsInline muted />
            <canvas ref={canvasRef} className="visually-hidden" />
            <div className="camera-actions">
              <button type="button" onClick={closeCamera}>
                Отмена
              </button>
              <button type="button" className="camera-shot-button" onClick={capturePhoto}>
                Сделать снимок
              </button>
            </div>
          </div>
        </div>
      )}

      <section className="results-panel" ref={resultsPanelRef}>
        <ResultToolbar
          hasResults={Boolean(response && bestMatch)}
          hasSimilar={similarMatches.length > 0}
          isSelected={Boolean(selectedMatch)}
          onBack={() => setSelectedMatch(null)}
          onNewSearch={resetSearch}
        />

        {!response && !isLoading && (
          <div className="empty-state">
            <h2>Загрузите фото, чтобы увидеть найденное вино</h2>
            <p>После поиска здесь появится большая карточка лучшего совпадения и похожие варианты из каталога.</p>
          </div>
        )}

        {isLoading && (
          <div className="empty-state">
            <div className="loader" />
            <h2>Ищем вино</h2>
            <p>Обрабатываем фото и сравниваем его с каталогом.</p>
          </div>
        )}

        {response && bestMatch && (
          <div className="search-results">
            {selectedMatch ? (
              <WineDetails match={selectedMatch} mode="selected" />
            ) : (
              <WineDetails match={bestMatch} />
            )}

            {!selectedMatch && similarMatches.length > 0 && (
              <section className="similar-section" id="similar-results">
                <h2>Похожие варианты</h2>
                <div className="similar-grid">
                  {similarMatches.map((match) => (
                    <SimilarWineCard
                      key={match.wine.id}
                      match={match}
                      isLoading={selectedWineIdLoading === match.wine.id}
                      onSelect={() => openSimilarWine(match)}
                    />
                  ))}
                </div>
              </section>
            )}
          </div>
        )}

        {response && !bestMatch && (
          <div className="empty-state">
            <h2>Совпадений нет</h2>
            <p>Попробуйте фото с более крупной этикеткой или увеличьте лимит результатов.</p>
          </div>
        )}
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
