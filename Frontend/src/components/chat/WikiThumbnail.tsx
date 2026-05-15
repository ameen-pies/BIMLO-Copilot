import React from "react";

export const WikiThumbnail: React.FC<{ wikiUrl: string }> = ({ wikiUrl }) => {
  const [imgSrc, setImgSrc] = React.useState<string | null>(null);

  React.useEffect(() => {
    const match = wikiUrl.match(/\/wiki\/([^#?]+)/);
    if (!match) return;
    const title = decodeURIComponent(match[1]);
    const langMatch = wikiUrl.match(/^https?:\/\/([a-z]{2})\.wikipedia/);
    const lang = langMatch ? langMatch[1] : "en";

    fetch(
      `https://${lang}.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(title)}`
    )
      .then(r => r.json())
      .then(data => {
        const url = data?.thumbnail?.source || data?.originalimage?.source;
        if (url) setImgSrc(url);
      })
      .catch(() => {});
  }, [wikiUrl]);

  if (!imgSrc) return null;
  return (
    <img
      src={imgSrc}
      alt=""
      className="h-16 w-16 object-cover rounded-lg shrink-0 opacity-90"
    />
  );
};
