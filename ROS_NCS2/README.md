1. INSTALACIJA OPEN VINO TOOLKIT-a

Za instalaciju Open Vino toolkit-a, koji je potreban za rad s Intel Neural Compute Stick 2 (NCS2), potrebno je detaljno pratiti upute na sljedećoj web stranici (pogledaj napomene prije):

https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html#install-openvino (1)

NAPOMENA 1.1: instalacija je provjerena za Ubuntu 16.04, moguće je i da radi na Ubuntu 18.04, ali nije 100% sigurno. 
NAPOMENA 1.2: ukoliko će se kojim slučajem raditi na virtualnoj mašini, kako bi uspjela prepoznati NCS2, potrebno je prethodno instalaciji Open Vino toolkita poduzeti dodatne korake koji su opisani na sljedećoj web stranici:

https://movidius.github.io/ncsdk/vm_config.html (2)

2. POKRETANJE YOLOV3 MODELA

Nakon što je instalacija uspješno izvršena te prikladno testirana kao što je opisano na (1) može se krenuti na proces pretvaranja Yolov3 mreže implementirane u Tensorflowu u oblik prikladan za pokretanje i izvršavanje na NCS2. Potrebno je pratiti upute na sljedećoj web stranici uz pojedine preinake (pogledaj napomenu 2.1 prije):

https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_YOLO_From_Tensorflow.html (3)

preinake su sljedeće:
kod prevođenja tensorflow modela (pb datoteka generirana nakon izvršavanja koraka 5 u (3)) u IR (Intermediate Representation) oblik naredba koju su oni naveli je:

python3 mo_tf.py --input_model /path/to/yolo_v3.pb --tensorflow_use_custom_operations_config $MO_ROOT/extensions/front/tf/yolo_v3.json

umjesto te naredbe pokrenuti sljedeću (pogledaj prije napomene 2.2 i 2.3):

python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model [path do pb datoteke]/frozen_darknet_yolov3_model.pb --tensorflow_use_custom_operations_config [path do json datoteke]/yolo_v3_changed.json --input_shape '(1, 416, 416, 3)' --data_type FP16

naredba navedena poviše je prikladna za izvođenje na NCS2.

NAPOMENA 2.1: prilikom pokretanja naredbe za konvertiranje (5. korak u (3)) iznimno je važno to napraviti s Tensorflow verzijom <= 1.12 instaliranom. Ukoliko se ta naredba pokrene s verzijom 1.13 konverzija parametara neće biti ispravna.
NAPOMENA 2.2: prilikom prevođenja modela u IR oblik koristiti json datoteku koja se nalazi ovdje:

https://github.com/PINTO0309/OpenVINO-YoloV3/blob/master/yolo_v3_changed.json (4)

možda se može koristiti i json datoteka koju su oni pružili, ali nije testirano.

NAPOMENA 2.3: kod pokretanja python skripte mo_tf.py koja služi za pretvaranje modela u IR oblik dobro je pokrenuti ju s -h zastavicom kako bi se vidjele mogućnosti prisutne kod konvertiranja (npr. zamjena RGB na BGR kanale ili skaliranje ulaznih vrijednosti) dodatne mogućnosti nisu testirane, sve je radilo ok i bez njih, jedine mogućnosti koje su korištene (i koje su nužne) su one koje su navedene u modificiranoj naredbi. Dodatne informacije su mogu pronaći na:

https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html (5)

3. TESTIRANJE TENSORFLOW MODELA KONVERTIRANOG U OBLIK POGODAN ZA IZVOĐENJE NA NCS2

Model se može testirati (C++) prateći upute na sljedećoj web stranici (pogledaj napomene prije):

https://docs.openvinotoolkit.org/latest/_inference_engine_samples_object_detection_demo_yolov3_async_README.html (6)

NAPOMENA 3.1: C++ demo za testiranje je potrebno buildati. Upute za buildanje na Linuxu se mogu pronaći pri dnu sljedeće stranice:

https://docs.openvinotoolkit.org/latest/_docs_IE_DG_Samples_Overview.html (7)

NAPOMENA 3.2: jednostavnije je pokrenuti Python demo za testiranje (nije potrebno buildanje), upute se mogu pronaći na sljedećoj web stranici:

https://docs.openvinotoolkit.org/latest/_inference_engine_ie_bridges_python_sample_object_detection_demo_yolov3_async_README.html (8)

4. POVEZIVANJE ROS-a S NCS2

Kod za povezivanje ROS-a s NCS2 je napravljen po uzoru na Intelov demo koji se može pronaći u sljedećem direktoriju:

/opt/intel/openvino/inference_engine/samples/object_detection_demo_yolov3_async

Nakon kloniranja Github repo-a s kodom koji povezuje ROS s NCS2, za testiranje se mogu pokretnuti sljedeće naredbe (u različitim terminalima):

rosrun ros_intel_stick image_publisher [path do mp4 datoteke]/Test_video_for_Object_detection.mp4

rosrun ros_intel_stick image_subscriber [path do xml datoteke]/frozen_darknet_yolov3_model.xml

rostopic echo /object_detection

5. DODACI:

Korisni Github repozitoriji s Tensorflow implementacijom Yolov3 mreže:

https://github.com/mystic123/tensorflow-yolo-v3 (korišten u tutorijalu)

https://github.com/YunYang1994/tensorflow-yolov3 

https://github.com/PINTO0309/OpenVINO-YoloV3 

ima ih vjerojatno još, ovo su samo neki.

