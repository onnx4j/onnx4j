/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.onnx4j.opsets.operator.fields;

import org.onnx4j.model.graph.node.Attribute;
import org.onnx4j.model.graph.node.Attributes;
import org.onnx4j.opsets.operator.Field;

public class AttributeField<T_TENSOR> extends Field<T_TENSOR> {

	private boolean required;
	private String name;

	public AttributeField(Attributes attrs, String name, Class<? extends Attribute<T_TENSOR>> clzOfAttrDT,
			T_TENSOR defaultVal, boolean required) {
		super(attrs.getAttrValue(name, clzOfAttrDT, defaultVal));
		this.required = required;
		this.name = name;
		
		this.assertNotNull();
	}

	public void assertNotNull() {
		if (this.required && this.data == null) {
			throw new IllegalArgumentException(String.format("Atrribute named \"%s\" is required.", this.name));
		}
	}

}
