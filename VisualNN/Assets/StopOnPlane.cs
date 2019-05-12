using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class StopOnPlane : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    void OnCollisionEnter(Collision col)
    {
        if (col.gameObject.GetComponent<Rigidbody>())
        {
            Vector3 pos = col.gameObject.transform.position;
            pos.z = 0;
            col.gameObject.transform.position = pos;
            Rigidbody rb = col.gameObject.GetComponent<Rigidbody>();
            rb.constraints = RigidbodyConstraints.FreezeAll;
            Debug.Log("Character Hit");
        }
    }
}
