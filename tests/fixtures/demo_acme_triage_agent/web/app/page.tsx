import { TriageQueue } from "./TriageQueue";

export default function AdminHome() {
  return (
    <main className="p-8">
      <h1 className="text-3xl font-bold mb-6">
        ACME Seguros — Painel de Triagem
      </h1>
      <p className="text-sm text-gray-600 mb-4">
        Sinistros recebidos hoje (CPFs e nomes redigidos para acesso operacional).
      </p>
      <TriageQueue />
    </main>
  );
}
