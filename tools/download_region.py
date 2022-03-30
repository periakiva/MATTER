import os
import json
import argparse
from glob import glob
from time import time
from copy import deepcopy
from datetime import datetime, timedelta, date

import intake
import kwcoco
import rasterio
import numpy as np
from pystac_client import Client
from PIL import Image, ImageDraw, ImageFont


def set_random_seed(seed_num):
    if seed_num == -1:
        print("No random seed set.")
    else:
        np.random.seed(seed_num)


def subsample_images(search_items, strategy, max_num_images, start_date, end_date):

    # Get times for all images
    datetimes = []
    for item_info in search_items.items:
        date = item_info.properties["datetime"][:10]
        time = item_info.properties["datetime"][11:-1]
        year = int(date[:4])
        month = int(date[5:7])
        day = int(date[8:10])
        hour = int(time[:2])
        minute = int(time[3:5])
        second = int(time[6:8])
        datetimes.append(datetime(year, month, day, hour, minute, second))

    # Sort items in ascending order according to capture time
    indices = np.arange(len(datetimes))
    indices = [index for _, index in sorted(zip(datetimes, indices))]
    datetimes = sorted(datetimes)

    if strategy == "equal":
        # Remove images by equalizing distribution of dates over capture range
        start_date = datetime.strptime(start_date, "%d/%m/%Y")
        end_date = datetime.strptime(end_date, "%d/%m/%Y")
        sub_search_items = deepcopy(search_items.items)

        # Project datetime value between [0-1]
        date_diff = (end_date - start_date).days
        date_diffs = [(date_diff - (end_date - date).days) / date_diff for date in datetimes]

        num_images = len(date_diffs)
        while num_images > max_num_images:
            # Recompute index values
            indices = np.arange(len(date_diffs))

            # Compute histogram of date deltas
            hist_dist, bin_bounds = np.histogram(date_diffs, bins=max_num_images, range=[0, 1])

            # Find the bin with the highest number of images
            hist_index = hist_dist.argmax()

            # Randomly select a capture from the bin with the largest bin
            min_bound, max_bound = bin_bounds[hist_index], bin_bounds[hist_index + 1]
            bool_arr = (min_bound <= date_diffs) & (max_bound >= date_diffs)
            assert hist_dist[hist_index] == bool_arr.sum()
            true_bin_indices = np.where(bool_arr)[0]
            bin_del_index = np.random.choice(true_bin_indices)
            del_index = indices[bin_del_index]

            # Remove capture from date_diffs, datetimes, and search_items
            del date_diffs[del_index]
            del datetimes[del_index]
            del sub_search_items[del_index]

            # Recompute the number of captures
            num_images = len(date_diffs)

    elif strategy == "first":
        # Get the first max_num_images captured
        sub_search_items = np.take(search_items.items, indices[:max_num_images])
    elif strategy == "last":
        # Get the last max_num_images captured
        sub_search_items = np.take(search_items.items, indices[-max_num_images:])
    else:
        raise NotImplementedError(f"Invalid subsampling strategy: {strategy}")

    # Overwrite items property in search_items variable
    assert type(sub_search_items) is list
    assert len(sub_search_items) == max_num_images
    search_items.items = sub_search_items

    return search_items


def load_base_kwcoco_file(base_dir):
    """Load or create kwcoco file if not found.

    Args:
        base_dir (str): A path to the base directory of unlabeled image dataset.

    Return:
        kwcoco_dataset (kwcoco.coco_dataset.CocoDataset): A dataset used to reference images and band transformations.
    """
    kwcoco_path = os.path.join(base_dir, "data.kwcoco.json")

    if os.path.isfile(kwcoco_path):
        # Load kwcoco file
        print(f"Loading kwcoco file from: {kwcoco_path}")
        kwcoco_dataset = kwcoco.CocoDataset(kwcoco_path)
    else:
        # Create kwcoco file
        print(f"Creating kwcoco file at: {kwcoco_path}")
        kwcoco_dataset = kwcoco.CocoDataset()

        text = json.dumps(kwcoco_dataset.dataset)
        with open(kwcoco_path, "w") as file:
            file.write(text)

        kwcoco_dataset.fpath = kwcoco_path

    return kwcoco_dataset


def create_gif_from_image_dirs(base_image_dir, datetimes):
    # Sort images based on save name
    image_dirs = sorted(glob(base_image_dir + "/*/"))

    assert len(image_dirs) == len(datetimes)

    # Get image timesteps
    dates = []
    for image_dir in image_dirs:
        raw_str = image_dir.split("/")[-2].split("_")[2]
        year = int(raw_str[:4])
        month = int(raw_str[4:6])
        day = int(raw_str[6:8])
        dates.append(datetime(year, month, day))

    # Sort image_dirs by date
    indices = np.arange(len(image_dirs))
    indices = [index for _, index in sorted(zip(dates, indices))]  # ordered based on dates
    image_dirs = np.take(image_dirs, indices)
    dates = np.take(dates, indices)

    rgb_images = []
    for image_dir in image_dirs:
        # Get RGB bands
        band_image_paths = sorted(glob(image_dir + "/*.tif"))

        r_band_path = band_image_paths[3]
        g_band_path = band_image_paths[2]
        b_band_path = band_image_paths[1]

        r_band = rasterio.open(r_band_path).read(1)
        g_band = rasterio.open(g_band_path).read(1)
        b_band = rasterio.open(b_band_path).read(1)

        # Create RGB image
        rgb_image = np.stack((r_band, g_band, b_band), axis=2)

        # Make rgb pretty
        rgb_image = np.clip(rgb_image, 0, 2500) / 2500
        gamma = 1.1
        rgb_image = rgb_image**(1 / gamma)
        rgb_image = rgb_image * 255
        rgb_image = rgb_image.astype("uint8")

        rgb_images.append(rgb_image)

    # format datetimes
    date_strs = []
    for dt in dates:
        date_str = dt.strftime("%m-%d-%Y")
        date_strs.append(date_str)

    # Create gif
    save_path = os.path.join(base_image_dir, "scene.gif")
    create_gif(rgb_images, save_path, image_text=date_strs)


def create_gif(image_list, save_path, fps=1, image_text=None, fontpct=5, overlay_images=None):
    """Create a gif image from a collection of numpy arrays.

    Args:
        image_list (list[numpy array]): A list of images in numpy format of type uint8.
        save_path (str): Path to save gif file.
        fps (float, optional): Frames per second. Defaults to 1.
        image_text (list[str], optional): A list of text to add to each frame of the gif.
            Must be the same length as iimage_list.
    """

    if len(image_list) < 2:
        print(f"Cannot create a GIF with less than 2 images, only {len(image_list)} provided.")
        return None
    elif len(image_list) == 2:
        img, imgs = Image.fromarray(image_list[0]), [Image.fromarray(image_list[1])]
    else:
        img, *imgs = [Image.fromarray(img) for img in image_list]

    if overlay_images is not None:
        assert len(overlay_images) == len(image_list)

        # Overlay images together
        images = [img]
        images.extend(imgs)

        images_comb = []
        for image_1, image_2 in zip(images, overlay_images):
            # Make sure images have alpha channel
            image_1.putalpha(1)
            image_2.putalpha(1)

            # Overlay images
            image_comb = Image.alpha_composite(image_1, image_2)
            images_comb.append(image_comb)

        img, *imgs = [img for img in images_comb]

    if image_text is not None:
        assert len(image_text) == len(image_list)

        # Have an issue loading larger font
        H = image_list[0].shape[0]
        if fontpct is None:
            font = ImageFont.load_default()
        else:
            if H < 200:
                font = ImageFont.load_default()
            else:
                fontsize = int(H * fontpct / 100)
                # Find fonts via "locate .ttf"
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf", fontsize)
                except:
                    print("Cannot find font at: /usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf")
                    font = ImageFont.load_default()

        images = [img]
        images.extend(imgs)
        for i, (img, text) in enumerate(zip(images, image_text)):
            draw = ImageDraw.Draw(img)
            draw.text((0, 0), text, (255, 0, 0), font=font)
            images[i] = img

        img, *imgs = images

    # Convert the images to higher quality
    images = [img]
    images.extend(imgs)
    img, *imgs = [img.quantize(dither=Image.NONE) for img in images]

    duration = int(1000 / fps)
    img.save(fp=save_path, format="GIF", append_images=imgs, save_all=True, duration=duration, loop=0, optimize=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Pull S2/L8 imagery from AWS catalog, download images, process images for ML methods.")
    parser.add_argument("download_dir", type=str, help="Path to directory where ")
    parser.add_argument(
        "change_region",
        type=str,
        choices=["change", "no_change"],
        help="Indicate whether you are expecting construction change is occuring in this region.",
    )
    parser.add_argument("--region_name", type=str, help="Give download folder a custom name instead of generated name.")
    parser.add_argument(
        "-gc",
        "--geo_coords",
        nargs="+",
        required=True,
        help="A list of geo-coordinates (lat-lon) corresponding to the query download region. Accepts either a 1 "
        " lat-lon pair (coordinate) or 4 lat-lon pairs (square). E.g. single coord: 40.6293_-74.8684. NOTE: Use "
        "quotes if any latitudes start with a negative and add a space after first quote ' -28.843058_136.439030'.",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="sentinel-s2-l2a-cogs",
        # choices=["sentinel-s2-l2a-cogs", "sentinel-s2-l1c"],
        help="Download this type of imagery.",
    )
    parser.add_argument(
        "-ds",
        "--start_date",
        type=str,
        default="01/01/2017",
        help="Beginning date to search for images of AOI. Format MM/DD/YYYY",
    )
    parser.add_argument(
        "-de",
        "--end_date",
        type=str,
        default="01/01/2020",
        help="End date to search for images of AOI. Format MM/DD/YYYY",
    )
    parser.add_argument("--crop_size", type=int, default=1098, help="Height and width of tile crops.")
    parser.add_argument(
        "--radius",
        type=float,
        default=0.001,
        help="The radius of the requested spatial region if only a single geo-coord was given.",
    )
    parser.add_argument(
        "-cc",
        "--max_cloud_cover",
        type=float,
        default=20,
        help="Maximum amount of allowable cloud coverage for a queried image.",
    )
    parser.add_argument("-dc",
                        "--min_data_coverage",
                        type=float,
                        default=80,
                        help="The allowable percentage of blank input images.")
    parser.add_argument("--min_num_images", type=int, default=2, help="Minimum number of images to process.")
    parser.add_argument("--max_num_images", type=int, default=100, help="Maximum number of images to process.")
    parser.add_argument(
        "--limit_num_image_strategy",
        type=str,
        default="equal",
        choices=["equal", "first"],
        help="Strategy on how to subsample images found in search.",
    )
    parser.add_argument(
        "--crop_pct",
        type=float,
        default=10,
        help="Number of crops that downloaded from total crops. We limit the total number to keep diversity high and "
        " save on disk space.",
    )
    parser.add_argument("--disable_gifs_gen",
                        default=False,
                        action="store_true",
                        help="Do not create GIFs of videos for this region.")
    parser.add_argument("--seed_num",
                        type=int,
                        default=0,
                        help="Random seed number, set to -1 if you want to disable this.")
    args = parser.parse_args()

    # Set random seed
    if args.seed_num != 0:
        print('WARNING: Changing the random seed number will give you different results than PEO dataset.')
    set_random_seed(args.seed_num)

    # Track time to download region
    start_time = time()

    # Check input arguments
    assert args.max_num_images > 0
    assert args.min_num_images > 0
    assert args.max_num_images > args.min_num_images
    assert (args.max_cloud_cover > 0) & (args.max_cloud_cover < 100)
    assert (args.min_data_coverage > 0) & (args.min_data_coverage < 100)

    # Create query from input arguments

    # Space range
    # Generate a bounding box based on number of coordinates given
    if len(args.geo_coords) == 1:
        # A single lat-long pair
        lat, lon = args.geo_coords[0].split("_")
        lat, lon = float(lat), float(lon)

        # Create square from point
        lat_min, lat_max = lat - args.radius, lat + args.radius
        lon_min, lon_max = lon - args.radius, lon + args.radius

        # TODO: Check that lat lon correspond to x and y from api
        # My guess: x = lon and y = lat
        space_extent = [lon_min, lat_min, lon_max, lat_max]

    elif len(args.geo_coords) == 4:
        # Convert list of coords to numpy array
        geo_coords = np.array((4, 2))
        for i in range(4):
            lat, lon = args.geo_coords[i].split("_")
            lat, lon = float(lat), float(lon)
            geo_coords[i][0] = lat
            geo_coords[i][1] = lon

        # Get boundary points from coords
        lat_min, lat_max = geo_coords.min(axis=0), geo_coords.max(axis=0)
        lon_min, lon_max = geo_coords.min(axis=1), geo_coords.max(axis=1)
        space_extent = [lon_min, lat_min, lon_max, lat_max]

    else:
        raise NotImplementedError(f"Cannot handle list of geo-coordinates of length: {len(args.geo_coords)}")

    # Time range
    ## Format date arguments into time
    start_date = datetime.strptime(args.start_date, "%m/%d/%Y")
    end_date = datetime.strptime(args.end_date, "%m/%d/%Y")

    # Get STAC query response
    ## Check to make sure that there are enough images for this region
    catalog = Client.open(
        "https://earth-search.aws.element84.com/v0")  # TODO: Add option to change catalog via command line (medium)
    search = catalog.search(
        bbox=space_extent,
        datetime=(start_date, end_date),
        collections=args.collection,
        query={
            "eo:cloud_cover": {
                "lt": args.max_cloud_cover
            },
            "data_coverage": {
                "gt": args.min_data_coverage
            }
        },
    )

    num_matched_images = search.matched()
    if num_matched_images <= args.min_num_images:
        print(f"Not enough images for given space-region range, only {num_matched_images}/{args.min_num_images}.")
    else:
        print(f"Number of images found that meet criteria: {num_matched_images}")

        for img_info in search.get_all_items_as_dict()["features"]:
            try:
                dc = img_info["properties"]["data_coverage"]
            except KeyError:
                dc = None

            cc = img_info["properties"]["eo:cloud_cover"]

            date = img_info["properties"]["datetime"]
            print(f"Date: {date} | Cloud cover: {cc} | Data cover: {dc}")

    search_items = search.get_all_items()

    # Limit the number of images
    if len(search_items) > args.max_num_images:
        print(f"Subsampling the number of matched images from {len(search_items)} to {args.max_num_images}.")
        search_items = subsample_images(search_items, args.limit_num_image_strategy, args.max_num_images,
                                        args.start_date, args.end_date)

    # Download images
    ## Create a place to store downloaded imagery
    if args.region_name is None:
        breakpoint()
        # TODO: Implement naming structure based on lat_lon_start-date_end-date (minor)
        pass
    else:
        region_name = args.region_name
        os.makedirs(args.download_dir, exist_ok=True)
        save_dir = os.path.join(args.download_dir, args.region_name)

    # TODO: Figure out how to handle case when running the same query again (minor)
    print(f"Saving images to: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    # Save input arguments
    save_args_path = os.path.join(save_dir, "args.json")
    with open(save_args_path, "w") as f:
        args_dict = vars(args)
        json.dump(args_dict, f, indent=4)

    # Load base kwcoco file (data.kwcoco.json)
    # base_kwcoco = load_base_kwcoco_file(args.download_dir)

    # Create desired tiles
    # TODO: Handle different sized tiles (minor)
    # TODO: Figure out how to get just image information for geo-region and not entire tile (medium)
    scene_height, scene_width = 10980, 10980

    tile_coords = []
    for i in range(scene_height // args.crop_size):
        for j in range(scene_width // args.crop_size):
            if np.random.rand() <= args.crop_pct / 100:
                tile_coords.append([i * args.crop_size, j * args.crop_size])

    band_ratios = {
        "B01": 6,
        "B02": 1,
        "B03": 1,
        "B04": 1,
        "B05": 2,
        "B06": 2,
        "B07": 2,
        "B08": 1,
        "B8A": 2,
        "B09": 6,
        "B10": 6,
        "B11": 2,
        "B12": 2,
    }

    # Create a kwcoco subdataset for each region
    sub_kwcoco = kwcoco.CocoDataset()

    # Download image from desired tiles

    if args.collection != "sentinel-s2-l1c-cog":
        os.environ["AWS_REQUEST_PAYER"] = "requester"
        os.environ["profile"] = "iarpa"

    band_names = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
    for i, crop_coord in enumerate(tile_coords):
        # Create crop folder to save images
        crop_dir = os.path.join(save_dir, f"crop_{str(i).zfill(3)}")
        os.makedirs(crop_dir, exist_ok=True)

        # Create video for sub-kwcoco dataset
        video_name = region_name
        vid_id = sub_kwcoco.add_video(video_name + "_" + str(i).zfill(4))

        datetimes = []
        for ii, item in enumerate(search_items):
            print(f"Tile [{i}/{len(tile_coords)}]: Image [{ii}/{len(search_items)}]")

            # Create time instance dir
            image_name = item.properties["sentinel:product_id"]
            image_dir = os.path.join(crop_dir, image_name)
            os.makedirs(image_dir, exist_ok=True)

            # Create a catalog object
            catalog = intake.open_stac_item(item)

            aux_data = []
            for band_name in band_names:
                # Get specific crop slices
                h_crop_slice = slice(crop_coord[0] // band_ratios[band_name],
                                     (crop_coord[0] + args.crop_size) // band_ratios[band_name])
                w_crop_slice = slice(crop_coord[1] // band_ratios[band_name],
                                     (crop_coord[1] + args.crop_size) // band_ratios[band_name])

                # Tile the image and load to disk
                da = catalog[band_name](chunks=dict(
                    band=1, y=args.crop_size // band_ratios[band_name], x=args.crop_size //
                    band_ratios[band_name])).to_dask()

                # Slice crop from desired tile
                crop = da[:, h_crop_slice, w_crop_slice][0]

                # Save band image
                save_path = os.path.join(image_dir, f"{band_name}.tif")

                # TODO: Add transform and crs to image metadata (medium)
                # https://rasterio.readthedocs.io/en/latest/quickstart.html#opening-a-dataset-in-writing-mode
                new_ds = rasterio.open(save_path,
                                       "w",
                                       driver="GTiff",
                                       height=crop.shape[0],
                                       width=crop.shape[1],
                                       count=1,
                                       dtype=crop.dtype)
                new_ds.write(crop, 1)

                # Add metadata to band_info
                band_info = {}
                band_info["height"] = crop.shape[0]
                band_info["width"] = crop.shape[1]
                band_info["crop_coords"] = [
                    [h_crop_slice.start, w_crop_slice.start],
                    [h_crop_slice.start, w_crop_slice.stop],
                    [h_crop_slice.stop, w_crop_slice.start],
                    [h_crop_slice.stop, w_crop_slice.stop],
                ]  # Not necessary but might be helpful
                band_info["channels"] = band_name
                band_info[
                    "file_name"] = save_path  # f"{band_name}.tif"  # relative path to image path (in this case the image dir)
                band_info["warp_aux_to_img"] = {
                    "type": "affine",
                    "scale": [band_ratios[band_name], band_ratios[band_name]],
                }  # Assume that the bands do not need to be orthorectificed
                aux_data.append(band_info)

            # Add image (all bands) to video with metadata
            sub_kwcoco.add_image(
                name="_".join(
                    (region_name, f"crop_{str(i).zfill(4)}", item.datetime.isoformat(), f"index_{str(ii).zfill(4)}")),
                file_name=image_dir,  # TODO: Check if giving a dir instead of path works (minor)
                height=args.crop_size,
                width=args.crop_size,
                channels="|".join(band_names),
                auxiliary=aux_data,
                video_id=vid_id,
                # timestamp=item.datetime.isoformat(),
                frame_index=ii,
                align_method="affine_warp",
            )

            datetimes.append(item.datetime)

        # TODO: Create gif of scenes (MAJOR)
        # create_gif_from_kwcoco_video(sub_kwcoco, vid_id, crop_dir)
        create_gif_from_image_dirs(crop_dir, datetimes)

        # TODO: Connect sub-kwcoco dataset to base kwcoco dataset (MAJOR)
        # breakpoint()
        # pass

    # Dump sub-kwcoco dataset
    sub_kwcoco_path = os.path.join(save_dir, "subdata.kwcoco.json")
    sub_kwcoco.fpath = sub_kwcoco_path
    text = json.dumps(sub_kwcoco.dataset)
    with open(sub_kwcoco_path, "w") as file:
        file.write(text)

    # TODO: Download DEM of region (medium)

    end_time = time()
    total_time = end_time - start_time
    print(f"Total time ellapsed: {str(timedelta(seconds=total_time))} H:M:S")

    # TODO: Save useful log data from region (minor)