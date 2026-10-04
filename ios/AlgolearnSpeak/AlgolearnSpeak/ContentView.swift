import SwiftUI

struct ContentView: View {
    @Bindable var link: PhoneLink

    var body: some View {
        VStack(spacing: 20) {
            hostRow
            Button(action: { link.wantsConnection ? link.disconnect() : link.connect() }) {
                Text(link.wantsConnection ? "Disconnect" : "Connect")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .tint(link.wantsConnection ? .red : .accentColor)

            badge
            levelMeter
            eventLog
            Spacer(minLength: 0)
        }
        .padding()
    }

    private var hostRow: some View {
        HStack {
            Text("Server").foregroundStyle(.secondary)
            TextField("192.168.86.188", text: $link.host)
                .textFieldStyle(.roundedBorder)
                .keyboardType(.numbersAndPunctuation)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .disabled(link.wantsConnection)
                .accessibilityLabel("Server address")
        }
    }

    private var badgeText: String {
        switch link.connection {
        case .disconnected: link.wantsConnection ? "Reconnecting" : "Offline"
        case .connecting: "Connecting"
        case .connected:
            switch link.serverState {
            case .idle: "Idle"
            case .speaking: "Speaking"
            case .listening: "Listening"
            case .processing: "Processing"
            }
        }
    }

    private var badgeColor: Color {
        guard link.connection == .connected else { return .gray }
        switch link.serverState {
        case .idle: return .secondary
        case .speaking: return .blue
        case .listening: return .green
        case .processing: return .orange
        }
    }

    private var badge: some View {
        Text(badgeText)
            .font(.system(size: 40, weight: .bold, design: .rounded))
            .minimumScaleFactor(0.6)
            .frame(maxWidth: .infinity, minHeight: 140)
            .foregroundStyle(.white)
            .background(badgeColor.gradient, in: RoundedRectangle(cornerRadius: 24))
            .accessibilityLabel("Status: \(badgeText)")
    }

    private var levelMeter: some View {
        let level = link.micOpen ? Double(Self.meterFraction(link.micLevel)) : 0
        return GeometryReader { geo in
            ZStack(alignment: .leading) {
                Capsule().fill(.quaternary)
                Capsule().fill(.green).frame(width: geo.size.width * level)
            }
        }
        .frame(height: 14)
        .opacity(link.micOpen ? 1 : 0.3)
        .animation(.linear(duration: 0.05), value: level)
        .accessibilityLabel("Microphone level")
        .accessibilityValue(link.micOpen ? "\(Int(level * 100)) percent" : "microphone closed")
    }

    /// Maps RMS (0...1) to a 0...1 meter over a 60 dB range.
    static func meterFraction(_ rms: Float) -> Float {
        guard rms > 0 else { return 0 }
        return max(0, min(1, (20 * log10(rms) + 60) / 60))
    }

    private var eventLog: some View {
        VStack(alignment: .leading, spacing: 4) {
            ForEach(Array(link.events.enumerated()), id: \.offset) { _, line in
                Text(line)
                    .font(.system(.footnote, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

#Preview { ContentView(link: PhoneLink()) }
