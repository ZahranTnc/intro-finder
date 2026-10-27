"""
tiktok.py — scrape TikTok accounts, download intros, transcribe, build corpus.

Usage:
    python tiktok.py --accounts @grmz.edits,@locortiz6,@fast.ae
    python tiktok.py --accounts @grmz.edits --max 50  # cap at 50 recent per account
    python tiktok.py --refresh                        # fetch new videos, keep old

Outputs:
    corpus/<acc>/<video_id>.mp4
    corpus/<acc>/<video_id>.json  (metadata: id, url, title, duration, views, likes)
    corpus/intros.json             (aggregate: all transcripts + view_rate scores)
"""
import argparse, json, re, sys, time
from pathlib import Path
from collections import defaultdict

def list_videos_playwright(handle: str, max_videos: int = 200) -> list[dict]:
    """Scroll profile page, collect video metadata from network responses."""
    from playwright.sync_api import sync_playwright
    
    videos = []
    seen = set()
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1280, "height": 720},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()
        
        def intercept(response):
            url = response.url
            if "/api/post/item_list" in url or "/api/user/detail" in url:
                try:
                    if response.status != 200:
                        return
                    data = response.json()
                    items = data.get("itemList") or []
                    for item in items:
                        vid_id = item.get("id")
                        if not vid_id or vid_id in seen:
                            continue
                        seen.add(vid_id)
                        stats = item.get("stats") or {}
                        video_info = item.get("video") or {}
                        videos.append({
                            "id": vid_id,
                            "url": f"https://www.tiktok.com/@{handle}/video/{vid_id}",
                            "title": (item.get("desc") or "").strip(),
                            "duration": video_info.get("duration") or video_info.get("durationSecond"),
                            "views": stats.get("playCount") or 0,
                            "likes": stats.get("diggCount") or 0,
                        })
                except Exception as e:
                    pass
        
        page.on("response", intercept)
        
        print(f"  loading profile page...")
        try:
            page.goto(f"https://www.tiktok.com/@{handle}", wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(3000)  # let initial API calls finish
        except Exception as e:
            print(f"  page load error: {e}")
        
        # scroll to trigger more loads
        print(f"  scrolling to load videos...")
        for scroll_n in range(15):
            if len(videos) >= max_videos:
                break
            page.evaluate("window.scrollBy(0, 800)")
            page.wait_for_timeout(1200)
            if scroll_n % 3 == 0:
                print(f"    {len(videos)} videos so far...")
        
        context.close()
        browser.close()
    
    return videos[:max_videos]


def download_video(url: str, out_path: str) -> bool:
    """Download one video with yt-dlp."""
    import yt_dlp
    opts = {
        "outtmpl": out_path,
        "format": "best[ext=mp4]",
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as y:
            y.download([url])
        return True
    except Exception as e:
        print(f"  [FAIL] {url}: {e}")
        return False


def transcribe_video(mp4_path: str) -> str:
    """Transcribe with faster-whisper, return text."""
    from faster_whisper import WhisperModel
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segs, _ = model.transcribe(mp4_path, vad_filter=True, beam_size=1, language="en")
    text = " ".join(s.text.strip() for s in segs).strip()
    return text


def build_corpus(accounts: list[str], max_per_account: int, refresh: bool):
    corpus_root = Path("corpus")
    corpus_root.mkdir(exist_ok=True)
    
    all_intros = []
    acc_medians = {}
    
    for handle in accounts:
        handle = handle.lstrip("@")
        acc_dir = corpus_root / handle
        acc_dir.mkdir(exist_ok=True)
        
        print(f"\n=== {handle} ===")
        print("listing videos...")
        videos = list_videos_playwright(handle, max_per_account)
        print(f"found {len(videos)} videos")
        
        views = [v["views"] for v in videos if v["views"]]
        acc_medians[handle] = sorted(views)[len(views)//2] if views else 1
        
        for i, v in enumerate(videos, 1):
            vid_id = v["id"]
            meta_file = acc_dir / f"{vid_id}.json"
            mp4_file = acc_dir / f"{vid_id}.mp4"
            
            if refresh and meta_file.exists():
                continue
            
            print(f"  [{i}/{len(videos)}] {vid_id} {v['views']:,} views")
            
            # download
            if not mp4_file.exists():
                ok = download_video(v["url"], str(mp4_file))
                if not ok:
                    continue
            
            # transcribe
            text = transcribe_video(str(mp4_file))
            
            # save meta
            v["text"] = text
            meta_file.write_text(json.dumps(v, ensure_ascii=False, indent=1), encoding="utf-8")
            all_intros.append({**v, "account": handle})
    
    # compute view_rate per intro
    for intro in all_intros:
        med = acc_medians.get(intro["account"], 1)
        intro["view_rate"] = intro["views"] / max(1, med)
    
    # write intros.json
    intros_file = corpus_root / "intros.json"
    intros_file.write_text(json.dumps(all_intros, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nDone: {len(all_intros)} intros -> {intros_file}")


def main():
    p = argparse.ArgumentParser(description="scrape TikTok accounts for movie intro corpus")
    p.add_argument("--accounts", required=True, help="comma-separated handles: @acc1,@acc2")
    p.add_argument("--max", type=int, default=200, help="max videos per account (default 200)")
    p.add_argument("--refresh", action="store_true", help="fetch new only, keep existing")
    args = p.parse_args()
    
    accounts = [a.strip() for a in args.accounts.split(",")]
    build_corpus(accounts, args.max, args.refresh)


if __name__ == "__main__":
    main()
