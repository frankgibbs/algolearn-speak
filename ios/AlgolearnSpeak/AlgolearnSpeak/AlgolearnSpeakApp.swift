import SwiftUI

@main
struct AlgolearnSpeakApp: App {
    @State private var link = PhoneLink()
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            ContentView(link: link)
                .onChange(of: scenePhase) { _, phase in
                    link.setForeground(phase == .active)
                }
        }
    }
}
