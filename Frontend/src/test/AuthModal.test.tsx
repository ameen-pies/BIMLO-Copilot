import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import AuthModal from "@/components/AuthModal";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => {
      const map: Record<string, string> = {
        "auth.welcome_back": "Welcome back",
        "auth.create_account": "Create account",
        "auth.login": "Login",
        "auth.signup": "Sign Up",
        "auth.email": "Email",
        "auth.username": "Username",
        "auth.password": "Password",
        "auth.google_login": "Continue with Google",
        "auth.or_with_email": "or with email",
        "auth.email_placeholder": "you@example.com",
        "auth.username_placeholder": "how should we call you?",
        "auth.password_placeholder_signup": "min. 6 characters",
        "auth.password_placeholder_login": "your password",
        "auth.continue_as_guest": "continue as guest",
        "auth.no_history_saved": "(no history saved)",
        "auth.google_not_configured": "Google Sign-In not configured.",
        "auth.google_signin_failed": "Google sign-in failed.",
        "auth.google_signin_failed_try_again": "Google sign-in failed. Try again.",
        "validation.fill_all_fields": "Please fill in all fields.",
        "validation.choose_username": "Please choose a username.",
      };
      return map[key] || key;
    },
  }),
}));

describe("AuthModal", () => {
  it("renders login form by default", () => {
    render(<AuthModal open={true} onClose={() => {}} onSuccess={() => {}} />);

    expect(screen.getByText("Welcome back")).toBeTruthy();
    expect(screen.getAllByText("Login")).toHaveLength(2);
    expect(screen.getByText("Sign Up")).toBeTruthy();
    expect(screen.getByText("Continue with Google")).toBeTruthy();
    expect(screen.getByText("or with email")).toBeTruthy();
  });

  it("switches to signup tab", async () => {
    const user = userEvent.setup();
    render(<AuthModal open={true} onClose={() => {}} onSuccess={() => {}} />);

    const signupTab = screen.getByText("Sign Up");
    await user.click(signupTab);

    expect(screen.getAllByText("Create account")).toHaveLength(2);
    expect(screen.getByText("Username")).toBeTruthy();
  });

  it("shows error when submitting empty form", async () => {
    const user = userEvent.setup();
    render(<AuthModal open={true} onClose={() => {}} onSuccess={() => {}} />);

    const submitBtn = screen.getAllByText("Login")[1];
    await user.click(submitBtn);

    expect(screen.getByText("Please fill in all fields.")).toBeTruthy();
  });

  it("does not render when closed", () => {
    const { container } = render(
      <AuthModal open={false} onClose={() => {}} onSuccess={() => {}} />
    );
    expect(container.textContent).not.toContain("Welcome back");
  });
});
