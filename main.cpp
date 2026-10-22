// ============================================================================
// Лабораторна робота № 5. Варіант 19.
// Маятник під дією періодичної сили (SFML + CMake + vcpkg).
//
// Фізична модель: математичний маятник довжиною L, масою m з тертям у підвісі
// та зовнішньою періодичною силою F(t) = F0 * cos(Omega * t), що діє на тягарець
// у напрямку руху (по дотичній до траєкторії).
//
// Рівняння руху (кут theta відлічується від вертикалі):
//     theta'' = -(g / L) * sin(theta) - gamma * theta' + (F0 / (m * L)) * cos(Omega * t)
//
// Програма складається з трьох частин:
//     handleInput()   - оброблення введення (клавіатура);
//     updatePhysics() - фізична модель (чисельне інтегрування методом Рунге-Кутти 4);
//     drawScene()     - відображення сцени, графіка theta(t) та параметрів.
//
// Керування:
//     Space       - пуск / пауза            R - перезапуск
//     Left/Right  - частота впливу Omega     (крок 0.1 рад/с)
//     Up/Down     - амплітуда сили F0        (крок 0.25 Н)
//     Esc         - вихід
// ============================================================================

#include <SFML/Graphics.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deque>
#include <optional>
#include <string>
#include <utility>

namespace
{
    const double PI = 3.14159265358979323846;

    // ------------------------------------------------------------------------
    // Параметри та стан фізичної моделі
    // ------------------------------------------------------------------------
    struct Params
    {
        double g     = 9.81;    // прискорення вільного падіння, м/с^2
        double L     = 1.0;     // довжина підвісу, м
        double m     = 1.0;     // маса тягарця, кг
        double gamma = 0.5;     // коефіцієнт згасання, 1/с
        double F0    = 1.0;     // амплітуда зовнішньої сили, Н        (змінюється з клавіатури)
        double Omega = 2.0;     // кутова частота зовнішньої сили, рад/с (змінюється з клавіатури)
    };

    struct State
    {
        double t     = 0.0;     // час моделювання, с
        double theta = 0.3;     // кут відхилення, рад
        double omega = 0.0;     // кутова швидкість, рад/с
    };

    // Кутове прискорення theta'' для заданих t, theta, omega
    double acceleration(const Params& p, double t, double theta, double omega)
    {
        return -(p.g / p.L) * std::sin(theta)
               - p.gamma * omega
               + (p.F0 / (p.m * p.L)) * std::cos(p.Omega * t);
    }

    // Один крок методу Рунге-Кутти 4-го порядку з кроком dt
    void rk4Step(const Params& p, State& s, double dt)
    {
        const double t = s.t, th = s.theta, w = s.omega;

        const double k1t = w;
        const double k1w = acceleration(p, t, th, w);

        const double k2t = w + 0.5 * dt * k1w;
        const double k2w = acceleration(p, t + 0.5 * dt, th + 0.5 * dt * k1t, w + 0.5 * dt * k1w);

        const double k3t = w + 0.5 * dt * k2w;
        const double k3w = acceleration(p, t + 0.5 * dt, th + 0.5 * dt * k2t, w + 0.5 * dt * k2w);

        const double k4t = w + dt * k3w;
        const double k4w = acceleration(p, t + dt, th + dt * k3t, w + dt * k3w);

        s.theta += dt / 6.0 * (k1t + 2 * k2t + 2 * k3t + k4t);
        s.omega += dt / 6.0 * (k1w + 2 * k2w + 2 * k3w + k4w);
        s.t     += dt;
    }

    // ------------------------------------------------------------------------
    // Стан програми
    // ------------------------------------------------------------------------
    struct App
    {
        Params p;
        State  s;
        bool   running = true;
        std::deque<std::pair<double, double>> history;   // (t, theta) для графіка
        double nextSample = 0.0;
    };

    const double PHYS_DT      = 0.002;   // крок інтегрування, с
    const double SAMPLE_DT    = 0.02;    // крок запису точок графіка, с
    const double GRAPH_SPAN   = 12.0;    // тривалість графіка, с
    const double AMP_WINDOW   = 10.0;    // вікно оцінки амплітуди, с

    const float PIVOT_X = 330.f, PIVOT_Y = 230.f, ROD_PX = 190.f;
    const float GRAPH_X = 40.f,  GRAPH_Y = 500.f, GRAPH_W = 920.f, GRAPH_H = 170.f;

    sf::String U(const std::string& utf8)
    {
        return sf::String::fromUtf8(utf8.begin(), utf8.end());
    }

    void resetModel(App& a)
    {
        a.s = State();
        a.history.clear();
        a.nextSample = 0.0;
    }

    // ------------------------------------------------------------------------
    // 1. Оброблення введення
    // ------------------------------------------------------------------------
    void handleInput(const sf::Event::KeyPressed& key, App& a, sf::RenderWindow& window)
    {
        switch (key.code)
        {
        case sf::Keyboard::Key::Escape: window.close(); break;
        case sf::Keyboard::Key::Space:  a.running = !a.running; break;
        case sf::Keyboard::Key::R:      resetModel(a); break;
        case sf::Keyboard::Key::Right:  a.p.Omega = std::min(8.0, a.p.Omega + 0.1); break;
        case sf::Keyboard::Key::Left:   a.p.Omega = std::max(0.1, a.p.Omega - 0.1); break;
        case sf::Keyboard::Key::Up:     a.p.F0    = std::min(5.0, a.p.F0 + 0.25);   break;
        case sf::Keyboard::Key::Down:   a.p.F0    = std::max(0.0, a.p.F0 - 0.25);   break;
        default: break;
        }
    }

    // ------------------------------------------------------------------------
    // 2. Фізична модель: просуваємо стан на frameTime секунд фіксованими кроками
    // ------------------------------------------------------------------------
    void updatePhysics(App& a, double frameTime, double& accumulator)
    {
        if (!a.running)
            return;

        accumulator += std::min(frameTime, 0.1);          // захист від великих пауз
        while (accumulator >= PHYS_DT)
        {
            rk4Step(a.p, a.s, PHYS_DT);
            accumulator -= PHYS_DT;

            if (a.s.t >= a.nextSample)                    // запис точки графіка
            {
                a.history.emplace_back(a.s.t, a.s.theta);
                a.nextSample += SAMPLE_DT;
                while (!a.history.empty() && a.history.front().first < a.s.t - GRAPH_SPAN)
                    a.history.pop_front();
            }
        }
    }

    // Амплітуда: максимум |theta| за останні AMP_WINDOW секунд
    double measuredAmplitude(const App& a)
    {
        double amp = 0.0;
        for (const auto& pt : a.history)
            if (pt.first >= a.s.t - AMP_WINDOW)
                amp = std::max(amp, std::fabs(pt.second));
        return amp;
    }

    // ------------------------------------------------------------------------
    // 3. Відображення
    // ------------------------------------------------------------------------
    std::string fmt(const char* format, double v)
    {
        char buf[64];
        std::snprintf(buf, sizeof(buf), format, v);
        return buf;
    }

    void drawText(sf::RenderWindow& w, const sf::Font& f, const std::string& s,
                  float x, float y, unsigned size = 18, sf::Color c = sf::Color(230, 230, 235))
    {
        sf::Text t(f, U(s), size);
        t.setPosition({x, y});
        t.setFillColor(c);
        w.draw(t);
    }

    void drawScene(sf::RenderWindow& w, const sf::Font& font, const App& a)
    {
        w.clear(sf::Color(24, 28, 38));

        // --- Маятник --------------------------------------------------------
        const float th = (float)a.s.theta;
        const sf::Vector2f pivot(PIVOT_X, PIVOT_Y);
        const sf::Vector2f bobPos(PIVOT_X + ROD_PX * std::sin(th), PIVOT_Y + ROD_PX * std::cos(th));

        sf::RectangleShape beam({200.f, 8.f});               // кронштейн
        beam.setOrigin({100.f, 8.f});
        beam.setPosition(pivot);
        beam.setFillColor(sf::Color(120, 125, 140));
        w.draw(beam);

        sf::VertexArray rod(sf::PrimitiveType::Lines);       // стрижень
        rod.append(sf::Vertex{pivot,  sf::Color(200, 200, 210)});
        rod.append(sf::Vertex{bobPos, sf::Color(200, 200, 210)});
        w.draw(rod);

        // Вертикаль (положення рівноваги)
        sf::VertexArray axis(sf::PrimitiveType::Lines);
        axis.append(sf::Vertex{pivot, sf::Color(90, 95, 110)});
        axis.append(sf::Vertex{sf::Vector2f(PIVOT_X, PIVOT_Y + ROD_PX + 40.f), sf::Color(90, 95, 110)});
        w.draw(axis);

        // Зовнішня сила: стрілка по дотичній до траєкторії тягарця
        const double fNow = a.p.F0 * std::cos(a.p.Omega * a.s.t);
        const sf::Vector2f tangent(std::cos(th), -std::sin(th));
        const sf::Vector2f tip = bobPos + tangent * (float)(fNow * 35.0);
        sf::VertexArray arrow(sf::PrimitiveType::Lines);
        arrow.append(sf::Vertex{bobPos, sf::Color(255, 170, 40)});
        arrow.append(sf::Vertex{tip,    sf::Color(255, 170, 40)});
        w.draw(arrow);
        sf::CircleShape tipDot(4.f);
        tipDot.setOrigin({4.f, 4.f});
        tipDot.setPosition(tip);
        tipDot.setFillColor(sf::Color(255, 170, 40));
        w.draw(tipDot);

        sf::CircleShape bob(20.f);                           // тягарець
        bob.setOrigin({20.f, 20.f});
        bob.setPosition(bobPos);
        bob.setFillColor(sf::Color(90, 170, 255));
        bob.setOutlineThickness(2.f);
        bob.setOutlineColor(sf::Color::White);
        w.draw(bob);

        sf::CircleShape hinge(5.f);
        hinge.setOrigin({5.f, 5.f});
        hinge.setPosition(pivot);
        hinge.setFillColor(sf::Color(230, 230, 235));
        w.draw(hinge);

        drawText(w, font, "Оранжева стрілка \u2014 зовнішня сила F(t)", 40.f, 20.f, 15, sf::Color(255, 170, 40));

        // --- Панель параметрів ---------------------------------------------
        const double omega0 = std::sqrt(a.p.g / a.p.L);
        const double amp = measuredAmplitude(a);
        float y = 30.f;
        const float x = 660.f, dy = 27.f;
        drawText(w, font, "Маятник під дією періодичної сили", x, y, 20, sf::Color(120, 200, 255)); y += dy + 6;
        drawText(w, font, std::string("Стан: ") + (a.running ? "ідуть обчислення" : "ПАУЗА"), x, y,
                 18, a.running ? sf::Color(120, 230, 140) : sf::Color(255, 130, 110)); y += dy + 6;
        drawText(w, font, "Час t = " + fmt("%.2f", a.s.t) + " с", x, y); y += dy;
        drawText(w, font, "Кут \u03B8 = " + fmt("%.1f", a.s.theta * 180.0 / PI) + "\u00B0", x, y); y += dy;
        drawText(w, font, "Кутова швидк. \u03C9 = " + fmt("%.2f", a.s.omega) + " рад/с", x, y); y += dy + 6;
        drawText(w, font, "Частота впливу \u03A9 = " + fmt("%.1f", a.p.Omega) + " рад/с", x, y,
                 18, sf::Color(255, 220, 120)); y += dy;
        drawText(w, font, "Амплітуда сили F0 = " + fmt("%.2f", a.p.F0) + " Н", x, y,
                 18, sf::Color(255, 220, 120)); y += dy + 6;
        drawText(w, font, "Власна частота \u03C9\u2080 = " + fmt("%.2f", omega0) + " рад/с", x, y); y += dy;
        drawText(w, font, "Згасання \u03B3 = " + fmt("%.2f", a.p.gamma) + " 1/с", x, y); y += dy;
        drawText(w, font, "Амплітуда коливань = " + fmt("%.1f", amp * 180.0 / PI) + "\u00B0", x, y,
                 18, sf::Color(150, 255, 170)); y += dy + 10;
        drawText(w, font, "Space \u2014 пуск/пауза,  R \u2014 перезапуск", x, y, 15, sf::Color(170, 175, 190)); y += 22.f;
        drawText(w, font, "\u2190 \u2192 \u2014 частота \u03A9,  \u2191 \u2193 \u2014 амплітуда F0", x, y, 15, sf::Color(170, 175, 190));

        // --- Графік theta(t) -----------------------------------------------
        sf::RectangleShape frame({GRAPH_W, GRAPH_H});
        frame.setPosition({GRAPH_X, GRAPH_Y});
        frame.setFillColor(sf::Color(16, 19, 27));
        frame.setOutlineThickness(1.f);
        frame.setOutlineColor(sf::Color(100, 105, 120));
        w.draw(frame);

        const float midY = GRAPH_Y + GRAPH_H / 2;
        sf::VertexArray zero(sf::PrimitiveType::Lines);
        zero.append(sf::Vertex{sf::Vector2f(GRAPH_X, midY), sf::Color(80, 85, 100)});
        zero.append(sf::Vertex{sf::Vector2f(GRAPH_X + GRAPH_W, midY), sf::Color(80, 85, 100)});
        w.draw(zero);

        const double thMax = 2.0;                            // межі графіка по осі \u03B8, рад
        sf::VertexArray curve(sf::PrimitiveType::LineStrip);
        const double t0 = a.s.t - GRAPH_SPAN;
        for (const auto& pt : a.history)
        {
            const float px = GRAPH_X + (float)((pt.first - t0) / GRAPH_SPAN) * GRAPH_W;
            double yy = midY - pt.second / thMax * (GRAPH_H / 2);
            yy = std::max((double)GRAPH_Y, std::min((double)(GRAPH_Y + GRAPH_H), yy));
            curve.append(sf::Vertex{sf::Vector2f(std::max(px, GRAPH_X), (float)yy), sf::Color(90, 170, 255)});
        }
        w.draw(curve);

        drawText(w, font, "\u03B8(t), межі \u00B1" + fmt("%.0f", thMax) + " рад (останні 12 с)", GRAPH_X + 6.f, GRAPH_Y + 4.f, 14,
                 sf::Color(170, 175, 190));

        w.display();
    }
}

int main()
{
    sf::RenderWindow window(sf::VideoMode({1000, 700}), "Lab 5. Variant 19: Pendulum with periodic force");
    window.setFramerateLimit(60);

    // Шрифт із підтримкою кирилиці (є в кожній Windows)
    sf::Font font;
    if (!font.openFromFile("C:/Windows/Fonts/arial.ttf") &&
        !font.openFromFile("C:/Windows/Fonts/segoeui.ttf"))
        return 1;

    App app;
    sf::Clock clock;
    double accumulator = 0.0;

    while (window.isOpen())
    {
        while (const std::optional event = window.pollEvent())
        {
            if (event->is<sf::Event::Closed>())
                window.close();
            else if (const auto* key = event->getIf<sf::Event::KeyPressed>())
                handleInput(*key, app, window);
        }

        updatePhysics(app, clock.restart().asSeconds(), accumulator);
        drawScene(window, font, app);
    }
    return 0;
}
