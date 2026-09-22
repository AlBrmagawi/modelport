import { describe, expect, it } from "vitest";
import { bytes, number, parseEvent, terminal } from "./api";

describe("evidence formatting and replay", () => {
  it("preserves measured zero and distinguishes missing values", () => {
    expect(number(0)).toBe("0");
    expect(number(null)).toBe("—");
    expect(bytes(1024)).toBe("1.0 KiB");
  });
  it("parses durable SSE without treating heartbeats as progress", () => {
    expect(parseEvent(": heartbeat")).toBeNull();
    expect(
      parseEvent(
        'id: 4\nevent: stage\ndata: {"sequence":4,"kind":"stage","payload":{"stage":"validating"}}',
      )?.sequence,
    ).toBe(4);
  });
  it("keeps cancellation intent separate from completed cancellation", () => {
    expect(terminal("cancel_requested")).toBe(false);
    expect(terminal("cancelled")).toBe(true);
  });
});
