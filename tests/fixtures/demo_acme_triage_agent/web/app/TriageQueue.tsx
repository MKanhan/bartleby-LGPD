"use client";

import { useEffect, useState } from "react";

type Claim = {
  claim_id: string;
  cpf_redacted: string;
  nome: string;
  classificacao: string;
  requer_revisao_humana: boolean;
};

export function TriageQueue() {
  const [items, setItems] = useState<Claim[]>([]);

  useEffect(() => {
    fetch("/api/queue").then((r) => r.json()).then(setItems);
  }, []);

  return (
    <table className="w-full text-left">
      <thead>
        <tr>
          <th>ID</th>
          <th>CPF</th>
          <th>Nome</th>
          <th>Classificação</th>
          <th>Revisão humana</th>
        </tr>
      </thead>
      <tbody>
        {items.map((c) => (
          <tr key={c.claim_id}>
            <td>{c.claim_id}</td>
            <td>{c.cpf_redacted}</td>
            <td>{c.nome}</td>
            <td>{c.classificacao}</td>
            <td>{c.requer_revisao_humana ? "Sim" : "Não"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
