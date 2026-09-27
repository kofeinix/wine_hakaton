import { Clock3, Grape, Heart, MapPin, MessageCircle, Palette, ScanLine, Trophy, Warehouse } from "@lucide/vue";

const ICONS = {
  scans: ScanLine,
  regions: MapPin,
  styles: Palette,
  grapes: Grape,
  producers: Warehouse,
  social: MessageCircle,
  favorites: Heart,
  time: Clock3,
  meta: Trophy,
};

export function useAchievementFormat() {
  function categoryIcon(category) {
    return ICONS[category] || Trophy;
  }

  // ≥ 10% — целые, < 10% — один знак, < 0,1% — «меньше 0,1%»
  function formatShare(value) {
    if (!value) return "Пока ни у кого";
    if (value < 0.1) return "Меньше 0,1% пользователей";
    const text = value >= 10 ? Math.round(value).toString() : value.toFixed(1).replace(".", ",");
    return `Получили ${text}% пользователей`;
  }

  return { categoryIcon, formatShare };
}
