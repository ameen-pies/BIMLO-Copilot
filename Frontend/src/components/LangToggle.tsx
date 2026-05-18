import { useTranslation } from "react-i18next";
import { Languages } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
} from "@/components/ui/dropdown-menu";

const FrFlag = () => (
  <img src="/flag-for-flag-france-svgrepo-com.svg" alt="Français" className="h-3.5 w-5 rounded-[2px] shrink-0 object-cover" />
);

const EnFlag = () => (
  <img src="/united-kingdom-uk-svgrepo-com.svg" alt="English" className="h-3.5 w-5 rounded-[2px] shrink-0 object-cover" />
);

const TnFlag = () => (
  <img src="/flag-for-flag-tunisia-svgrepo-com.svg" alt="العربية" className="h-3.5 w-5 rounded-[2px] shrink-0 object-cover" />
);

const languages = [
  { code: "fr", label: "Français", Flag: FrFlag },
  { code: "en", label: "English", Flag: EnFlag },
  { code: "ar", label: "العربية", Flag: TnFlag },
];

const LangToggle = () => {
  const { t, i18n } = useTranslation();

  return (
    <DropdownMenu modal={false}>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          className="h-9 w-9 text-muted-foreground hover:text-foreground"
          aria-label={t("common.switch_language")}
        >
          <Languages className="h-5 w-5" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuRadioGroup
          value={i18n.language}
          onValueChange={(val) => i18n.changeLanguage(val)}
        >
          {languages.map((lang) => (
            <DropdownMenuRadioItem key={lang.code} value={lang.code}>
              <span className="flex items-center gap-2">
                <lang.Flag />
                {lang.label}
              </span>
            </DropdownMenuRadioItem>
          ))}
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  );
};

export default LangToggle;
